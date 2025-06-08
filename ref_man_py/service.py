from typing import Optional, cast
import json
import os
import time
from pathlib import Path
from threading import Thread
from multiprocessing import Process
import dataclasses

import yaml
import requests
import psutil
from flask import Flask, request, Response
from werkzeug import serving
import magic
import brotli

from common_pyutil.log import get_file_and_stream_logger, get_stream_logger

from s2cache import SemanticScholar
from s2cache.models import PaperData, PaperDetails, Citation, Error, References, Citations
from s2cache.util import dumps_json

from . import __version__
from .const import default_headers
from .util import (fetch_url_info, fetch_url_info_parallel,
                   parallel_fetch, post_json_wrapper, check_proxy_port,
                   filter_fields, dumps_data_or_error)
from .config import Config, default_fields
from .arxiv import arxiv_get, arxiv_fetch, arxiv_helper
from .dblp import dblp_helper
from .cvf import CVF
from .pdf_cache import PDFCache


app = Flask("RefMan")


class RefMan:
    """*ref-man* server for network requests.

    We use a separate python process for efficient (and sometimes parallel)
    fetching of network requests.

    Args:
        host: host on which to bind
        port: port on which to bind
        proxy_port: Port for the proxy server. Used by :code:`fetch_proxy`, usually for PDFs.
        proxy_everything: Whether to fetch all requests via proxy.
        proxy_everything_port: Port for the proxy server on which everything is proxied.
                               Used by supported methods.
        data_dir: Directory where the Semantic Scholar Cache is stored.
                  See :func:`load_ss_cache`
        config_dir: Directory to store configuration related files
        corpus_cache_dir: Directory of Semantic Scholar Citations Corpus
        local_pdfs_dir: Local directory where the pdf files are stored.
        remote_pdfs_dir: Remote directory where the pdf files are stored.
        remote_links_cache: File mapping local pdfs to remote links.
        batch_size: Number of parallel requests to send in case parallel requests is
                    implemented for that method.
        debug: Whether to start the service in debug mode
        logfile: Optional log file for logging
        logfile_verbosity: logfile verbosity level
        verbosity: stdiout verbosity level
        threaded: Start the flask server in threaded mode. Defaults to :code:`True`.

    :code:`remote_pdfs_dir` has to be an :code:`rclone` path and the pdf files from
    :code:`local_pdfs_dir` is synced to that with :code:`rclone`.

    """
    # TODO: Generate default config and place in config dir.
    #       As of now only SemanticScholar configuration is loaded if it exists
    # TODO: Also allow the user to add config options like request headers,
    #       proxy ports, data_dir etc.
    def __init__(self, *, host: str, port: int, proxy_port: int, proxy_everything: bool,
                 proxy_everything_port: int, data_dir: Path, corpus_cache_dir: Path,
                 config_dir: Path, local_pdfs_dir: Path,
                 remote_pdfs_dir: Path, remote_links_cache: Path,
                 batch_size: int, logfile: str, logdir: Path, logfile_verbosity: str,
                 debug: bool, verbosity: str, threaded: bool):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.corpus_cache_dir = corpus_cache_dir
        self.local_pdfs_dir = local_pdfs_dir
        self.remote_pdfs_dir = remote_pdfs_dir
        self.remote_links_cache = remote_links_cache
        self.proxy_port: Optional[int] = proxy_port
        self.proxy_everything: Optional[int] = proxy_everything
        self.proxy_everything_port: Optional[int] = proxy_everything_port
        if not self.config_dir.exists():
            os.makedirs(self.config_dir)
        self.debug = debug

        self.logfile = logfile
        self.logdir = logdir
        self.logfile_verbosity = logfile_verbosity
        self.verbosity = verbosity
        self.threaded = threaded
        self._requests_timeout = 60

        self._init_loggers()
        self._init_config()
        self._init_s2_and_fields()
        self._init_remote_cache()

        self.cvf_helper = CVF(self.config_dir, self.logger)

        # NOTE: Checks only once for the proxy, see util.check_proxy
        #       for a persistent solution
        self.check_proxies()
        self._app = app
        self.init_routes()

    def _init_config(self):
        self.config_file: Optional[Path] = self.config_dir.joinpath("config.yaml")
        self.config_file = self.config_file if self.config_file.exists() else None
        if self.config_file:
            self.logi(f"Loaded config file {self.config_file}")
            with open(self.config_file) as f:
                _config = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            _config = {"data": default_fields()}
        self.config = Config(**_config)

    def _init_s2_and_fields(self):
        self.s2 = SemanticScholar(config_or_file=self.config.s2,
                                  cache_dir=self.data_dir,
                                  citations_cache_dir=self.corpus_cache_dir,
                                  logger_name="ref-man")

        # fields are which are filtered and sent to interface.
        # Determined via config
        self._paper_fields = {x[0] if isinstance(x, list) else x: x
                              for x in self.config.data.details.fields}
        self._paper_fields.update({"references": "references",
                                   "citations": "citations"})
        self._search_fields = {x[0] if isinstance(x, list) else x: x
                               for x in self.config.data.search.fields}
        self._search_fields.update({"references": "references",
                                    "citations": "citations"})
        self._references_fields = {x[0] if isinstance(x, list) else x: x
                                   for x in self.config.data.references.fields}
        self._citations_fields = {x[0] if isinstance(x, list) else x: x
                                  for x in self.config.data.citations.fields}

    def _get(self, url: str, **kwargs) -> requests.Response:
        return requests.get(url, timeout=self._requests_timeout, **kwargs)


    def _filter_paper_subr(self, fields: dict[str, str | list[str]]):
        """Add default fields for :code:`fields` in case they're missing

        Args:
            fields: Is a mapping of :code:`field` name to a getter, which
                    can be a string or a list of strings to lookup a value
                    in a nested :code:`dict`.
        """
        paper_fields = fields.get("paper_fields", self._paper_fields)
        citations_fields = fields.get("citations_fields", self._citations_fields)
        references_fields = fields.get("references_fields", self._references_fields)
        return fields, paper_fields, citations_fields, references_fields

    def _filter_paper_details(self, data: PaperDetails, fields: dict)\
            -> PaperDetails:
        fields, paper_fields, citations_fields, references_fields =\
            self._filter_paper_subr(fields)
        _data = filter_fields(data, paper_fields)
        data = PaperDetails(**_data)
        if citations_fields != "all":
            data.citations = [PaperDetails(**filter_fields(x, citations_fields))
                              for x in filter(None, data.citations)]
        if references_fields != "all":
            data.references = [PaperDetails(**filter_fields(x, references_fields))
                               for x in filter(None, data.references)]
        return data

    def _filter_paper_data(self, data: PaperData, fields: dict)\
            -> PaperData:
        """Filter allowed fields and apply count limits to S2 data citations and references

        Args:
            data: S2 Data

        Limits are defined in configuration

        """
        fields, paper_fields, citations_fields, references_fields =\
            self._filter_paper_subr(fields)
        if paper_fields != "all":
            _details: dict = filter_fields(data.details, paper_fields)
            data.details = PaperDetails(**_details)
        data.citations.data = [x for x in data.citations.data if x.citingPaper]
        data.references.data = [x for x in data.references.data if x.citedPaper]
        if citations_fields != "all":
            for i, x in enumerate(data.citations.data):
                data.citations.data[i].citingPaper =\
                    PaperDetails(**filter_fields(x.citingPaper, citations_fields))
        if references_fields != "all":
            for i, y in enumerate(data.references.data):
                data.references.data[i].citedPaper =\
                    PaperDetails(**filter_fields(y.citedPaper, references_fields))
        return data

    def _init_remote_cache(self):
        """Initialize cache (and map) of remote and local pdf files.

        The files can be synced from and to an :code:`rclone` remote.

        """
        self.update_cache_run = None
        if self.local_pdfs_dir and self.remote_pdfs_dir and self.remote_links_cache:
            self.pdf_cache_helper: Optional[PDFCache] =\
                PDFCache(self.local_pdfs_dir, Path(self.remote_pdfs_dir),
                         Path(self.remote_links_cache), self.logger)
        else:
            self.pdf_cache_helper = None
            self.logger.warning("All arguments required for pdf cache not given.\n"
                                "Will not maintain remote pdf links cache.")

    def _init_loggers(self):
        verbosity_levels = {"info", "error", "debug"}
        self.verbosity = (self.verbosity in verbosity_levels and self.verbosity) or "info"
        self.logfile_verbosity = (self.logfile_verbosity in verbosity_levels and
                                  self.logfile_verbosity) or "debug"
        if self.logdir and self.logfile:
            _, self.logger = get_file_and_stream_logger(logger_name="ref-man",
                                                        logdir=str(self.logdir.absolute()),
                                                        log_file_name=self.logfile,
                                                        new_file=False,
                                                        file_log_level=self.logfile_verbosity,
                                                        stream_log_level=self.verbosity,
                                                        logger_level="debug")
            self.logger.info(f"Log file is {self.logdir}/{self.logfile}")
            self.logger.debug(f"File log level is set to {self.logfile_verbosity}")
            self.logger.debug(f"Stream log level is set to {self.verbosity}")
        else:
            self.logger = get_stream_logger("ref-man", log_level=self.verbosity)
            self.logger.debug(f"Log level is set to {self.verbosity}.")

    def logi(self, msg: str) -> str:
        self.logger.info(msg)
        return msg

    def logd(self, msg: str) -> str:
        self.logger.debug(msg)
        return msg

    def logw(self, msg: str) -> str:
        self.logger.warning(msg)
        return msg

    def loge(self, msg: str) -> str:
        self.logger.error(msg)
        return msg

    def check_proxies(self) -> str:
        """Check any proxies if given.

        Proxies may help bypass paywalls if they connect to your institute
        which has access to certain articles. You'll need to have a valid
        proxy server that connects to the network of your institute.

        """
        msgs: list[str] = []
        self.proxies = None
        self.everything_proxies = None
        if self.proxy_everything_port:
            status, msg, proxies = check_proxy_port(self.proxy_everything_port,
                                                    "proxy everything",
                                                    self.logger)
            msgs.append(msg)
            if status:
                msg = "Warning: proxy_everything is only implemented for DBLP."
                self.logger.warning(msg)
                msgs.append(msg)
                self.everything_proxies = proxies
        if self.proxy_port is not None:
            proxy_name = f"Proxy with port: {self.proxy_port}"
            status, msg, proxies = check_proxy_port(self.proxy_port,
                                                    proxy_name,
                                                    self.logger)
            msgs.append(msg)
            if status:
                self.proxies = proxies
        return "\n".join(msgs)

    # TODO: There are inconsistencies in how the methods return. Some do via json,
    #       and some text
    # TODO: This entire method should be rewritten so that the functions are
    #       outside. This just grew out of nowhere LOL (not really but...)
    def init_routes(self):
        def filter_paper_and_dump(data: Error | PaperData | PaperDetails, request) -> str:
            if isinstance(data, Error):
                return dumps_data_or_error(data)
            if request.method == "POST":
                post_data = request.json
                if not post_data:
                    return "ERROR"
                fields = post_data.get("fields", {})
            else:
                fields = {}
            if isinstance(data, PaperData):
                data = self._filter_paper_data(data, fields)
            else:
                data = self._filter_paper_details(data, fields)
            if fields:
                data = data.asdict()
            return dumps_data_or_error(data)

        @app.route("/arxiv", methods=["GET", "POST"])
        def arxiv():
            if request.method == "GET":
                if "id" in request.args:
                    id = request.args["id"]
                else:
                    return json.dumps("NO ID GIVEN")
                return arxiv_get(id)
            else:
                result = post_json_wrapper(request, arxiv_fetch, arxiv_helper,
                                           self.batch_size, "Arxiv", self.logger)
                return json.dumps(result)

        @app.route("/s2_paper", methods=["GET", "POST"])
        def s2_paper() -> str:
            if "id" in request.args:
                ID = request.args["id"]
            else:
                return json.dumps("NO ID GIVEN")
            if "id_type" in request.args:
                id_type = request.args["id_type"]
            else:
                return json.dumps("NO ID_TYPE GIVEN")
            force = bool(request.args.get("force", False))
            paper_data = bool(request.args.get("paper_data", False))
            local_only = bool(request.args.get("cache_only", False))
            if paper_data:
                data: Error | PaperData | PaperDetails = self.s2.get_data_for_id(
                    id_type, ID, force, local_only)
            else:
                data = self.s2.get_details_for_id(id_type, ID, force)
            return filter_paper_and_dump(data, request)

        @app.route("/s2_paper_local", methods=["GET", "POST"])
        def s2_paper_local() -> str:
            if "corpus_id" not in request.args and "ssid" not in request.args:
                return "BAD PARAMS"
            corpus_id = request.args.get("corpus_id", None)
            ssid = request.args.get("ssid", None)
            if corpus_id is not None:
                corpus_id = int(corpus_id)  # type: ignore
            data: Error | PaperData | PaperDetails = self.s2.get_data_from_corpus_cache(
                corpus_id=corpus_id, ssid=ssid)
            return filter_paper_and_dump(data, request)

        @app.route("/s2_get_updated_paper", methods=["GET"])
        def s2_update_paper():
            ID = request.args.get("id", None)
            if not ID:
                return json.dumps("NO ID GIVEN")
            keys = request.args.get("keys", None)
            if not keys:
                return json.dumps("NO KEYS TO UPDATE GIVEN")
            data = self.s2.update_and_fetch_paper(ID, keys.split(","))
            return filter_paper_and_dump(data, request)

        @app.route("/s2_corpus_id", methods=["GET"])
        def s2_corpus_id():
            if "id" in request.args:
                ID = request.args["id"]
            else:
                return json.dumps("NO ID GIVEN")
            if "id_type" in request.args:
                id_type = request.args["id_type"]
            else:
                return json.dumps("NO ID_TYPE GIVEN")
            data = self.s2.id_to_corpus_id(id_type, ID)
            return dumps_json(data)

        @app.route("/s2_config", methods=["GET"])
        def s2_config():
            return dumps_json(self.s2._config)

        @app.route("/s2_details/<ssid>", methods=["GET"])
        def s2_details(ssid: str) -> str | bytes:
            if "force" in request.args:
                force = True
            else:
                force = False
            details = self.s2.paper_details(ssid, force=force)
            if isinstance(details, Error):
                return dumps_data_or_error(details)
            details = self._filter_paper_details(details, {})
            return dumps_data_or_error(details)

        @app.route("/ensure_citations/<ssid>", methods=["POST"])
        def s2_ensure_citations(ssid: str) -> str | bytes:
            if "force" in request.args:
                force = True
            else:
                force = False
            details = self.s2.paper_details(ssid, force=force)
            if isinstance(details, Error):
                return dumps_data_or_error(details)
            details = self._filter_paper_details(details, {})
            return dumps_data_or_error(details)

        def filter_subr(values, key, fields):
            citetype = "citedPaper" if key == "references" else "citingPaper"
            if isinstance(values, (References, Citations)):
                if isinstance(values.data[0], dict):
                    return [filter_fields(x[citetype], fields) for x in values.data]
                return [filter_fields(dataclasses.asdict(x)[citetype], fields) for x in values.data]
            else:
                return [filter_fields(x, fields) for x in values]

        def s2_citations_references_subr(request, ssid: str, key) -> str | bytes:
            """Get requested citations or references for a paper.

            TODO: describe the filters and count mechanism

            Requires an :code:`ssid` for the paper. Optional :code:`count` and
            :code:`filters` can be given in the request as arguments.

            See :meth:`s2_citations_references_subr` for details

            """
            func = getattr(self.s2, key)
            if "count" in request.args:
                count = int(request.args["count"])
                if count > 10000:
                    return json.dumps("MAX 10000 CITATIONS CAN BE FETCHED AT ONCE.")
            else:
                count = None
            offset = int(request.args.get("offset", 0))
            fields = self._references_fields if key == "references"\
                else self._citations_fields
            if request.method == "GET":
                if "filters" in request.args:
                    return json.dumps("Filters only supported with POST")
                values = func(ssid, offset=offset, limit=count)
            else:
                if self.debug and hasattr(request, "json"):
                    print("REQUEST JSON", request.json)
                data = request.json
                if not data:
                    return json.dumps("METHOD NOT IMPLEMENTED without data")
                else:
                    filters = data.get("filters")
                    if filters:
                        func = self.s2.filter_citations if key == "citations" else\
                            self.s2.filter_references
                        values = func(ssid, filters=filters, num=count)
                    else:
                        values = func(ssid, offset=offset, limit=count)
                    fields = data.get("fields", fields)
            filtered = filter_subr(values, key, fields)
            return dumps_data_or_error(filtered)

        @app.route("/s2_citations/<ssid>", methods=["GET", "POST"])
        def s2_citations(ssid: str) -> str | bytes:
            """Get the citations for a paper from S2 graph api.

            Requires an :code:`ssid` for the paper. Optional :code:`count` and
            :code:`filters` can be given in the request as arguments.

            See :meth:`s2_citations_references_subr` for details

            """
            return s2_citations_references_subr(request, ssid, "citations")

        @app.route("/s2_references/<ssid>", methods=["GET", "POST"])
        def s2_references(ssid: str) -> str | bytes:
            """Get the references for a paper from S2 graph api.

            Requires an :code:`ssid` for the paper. Optional :code:`count` and
            :code:`filters` can be given in the request as arguments.

            See :meth:`s2_citations_references_subr` for details

            """
            return s2_citations_references_subr(request, ssid, "references")

        @app.route("/s2_next_citations/<ssid>", methods=["GET", "POST"])
        def s2_next_citations(ssid: str) -> str | bytes:
            """Get the next citations from S2 graph api.

            See :meth:`SemanticScholar.next_citations` for implementation details.
            """
            if "filters" in request.args:
                return json.dumps("FILTERS NOT SUPPORTED WITH GET")
            if "count" in request.args:
                count = int(request.args["count"])
                if count > 10000:
                    json.dumps("MAX 10000 CITATIONS CAN BE FETCHED AT ONCE.")
            else:
                count = 0
            if request.method == "GET":
                return dumps_data_or_error(self.s2.next_citations(ssid, count))
            else:
                data = request.json
                if not data:
                    return json.dumps("BAD PARAMS")
                fields = data.get("fields", "all")
                values = self.s2.next_citations(ssid, count)
                filtered = filter_subr(values, "citations", fields)
                return dumps_data_or_error(filtered)

        @app.route("/s2_citations_params", methods=["GET", "POST"])
        def s2_citations_params():
            """Get or set parameters for fetching citations from S2.

            The parameters can be queried or set dynamically.
            """
            def json_defaults(x):
                return str(x).replace("typing.", "").replace("<class \'", "").replace("\'>", "")
            filters = [(k, v.__annotations__) for k, v in self.s2.filters.items()],
            if request.method == "GET":
                return json.dumps({"count": "Number of citations to return",
                                   "filters": filters},
                                  default=json_defaults)
            else:
                # TODO: Not Implemented yet
                return json.dumps("METHOD NOT IMPLEMENTED")

        @app.route("/s2_recommendations", methods=["GET", "POST"])
        def recommendations():
            """Search Semantic Scholar for a query string via the graph api."""
            count = int(request.args.get("count", 0))
            if request.method == "GET":
                if "paperid" in request.args:
                    paperid = [request.args["paperid"]]
                else:
                    return json.dumps("NO paperid given")
                return self.s2.recommendations(paperid, [], count)
            else:
                data = request.json
                pos_ids = data and data.get("pos-ids", None)
                neg_ids = data and data.get("neg-ids", None)
                if not pos_ids:
                    return json.dumps("pos-ids not given")
                if not neg_ids:
                    return json.dumps("neg-ids not given")
                return self.s2.recommendations(pos_ids, neg_ids, count)

        @app.route("/s2_search", methods=["GET", "POST"])
        def s2_search():
            """Search Semantic Scholar for a query string via the graph api."""
            if request.method == "GET":
                if "q" in request.args:
                    query = request.args["q"]
                else:
                    return json.dumps("NO QUERY GIVEN")
                result = self.s2.search(query)
                if isinstance(result, dict):
                    data = result.get("data", [])
                    if data:
                        result["data"] = [filter_fields(x, self._search_fields) for x in data]
                    return result
                return result
            else:
                return json.dumps("METHOD NOT IMPLEMENTED")

        @app.route("/url_info", methods=["GET"])
        def url_info() -> str:
            """Fetch info about a given url or urls based on certain rules.

            See :func:`fetch_url_info` for details.
            """
            if "url" in request.args and request.args["url"]:
                url = request.args["url"]
                urls = None
            elif "urls" in request.args and request.args["urls"]:
                urls = request.args["urls"].split(",")
                url = None
            else:
                return json.dumps("NO URL or URLs GIVEN")
            if urls is not None:
                return parallel_fetch(urls, fetch_url_info_parallel, self.batch_size)
            elif url is not None:
                return json.dumps(fetch_url_info(url))
            else:
                return json.dumps("NO URL or URLs GIVEN")

        @app.route("/set_proxy")
        def set_proxy():
            """Set or unset proxy port"""
            if request.args.get("set"):
                temp = request.args.get("proxy_port", None)
                if temp is not None:
                    self.proxy_port = int(temp)
                temp = request.args.get("proxy_port", None)
                if temp:
                    self.proxy_everything_port = int(temp)
                return self.check_proxies()
            elif request.args.get("unset"):
                self.proxy_port = None
                self.proxy_everything_port = None
                self.proxies = None
                return "Unset all proxies"

        @app.route("/fetch_url")
        def fetch_url():
            """Fetch given URL.

            Optionally if :attr:`self.proxies` if :attr:`self.proxies` is
            not :code:`None` then fetch via those proxies

            """
            url = request.args.get("url")
            if not url:
                return json.dumps("NO URL GIVEN or BAD URL")
            noproxy = request.args.get("noproxy")
            keys = [*request.args.keys()]
            # NOTE: Rest of the keys are part of the URL
            if len(keys) > 1:
                url = url + "&" + "&".join([f"{k}={v}" for k, v in request.args.items()
                                            if k not in {"url", "noproxy"}])
            # DEBUG code
            # if url == "https://arxiv.org/pdf/2006.01912":
            #     with os.path.expanduser("~/pdf_file.pdf", "rb") as f:
            #         pdf_data = f.read()
            #     response = make_response(pdf_data)
            #     response.headers["Content-Type"] = "application/pdf"
            #     return response
            self.logger.debug(f"Fetching {url} with proxies {self.proxies}")
            if not noproxy and self.proxies:
                try:
                    response = self._get(url, headers=default_headers, proxies=self.proxies)
                except requests.exceptions.Timeout:
                    self.logger.error("Proxy not reachable. Fetching without proxy")
                    self.proxies = None
                    response = self._get(url, headers=default_headers)
                except requests.exceptions.ProxyError:
                    self.logger.error("Proxy not reachable. Fetching without proxy")
                    self.proxies = None
                    response = self._get(url, headers=default_headers)
            else:
                if not noproxy:
                    self.logger.warning("Proxy dead. Fetching without proxy")
                response = self._get(url, headers=default_headers)
            if magic.detect_from_content(response.content).mime_type == "application/octet-stream":
                try:
                    content = brotli.decompress(response.content)
                    response.content = content
                except Exception:
                    pass
            if url.startswith("http:") or response.url.startswith("https:"):
                return Response(response.content)
            elif response.url != url:
                if response.headers["Content-Type"] in\
                   {"application/pdf", "application/octet-stream"}:
                    return Response(response.content)
                elif response.headers["Content-Type"].startswith("text"):
                    return json.dumps({"redirect": response.url,
                                       "content": response.content.decode("utf-8")})
                else:
                    return json.dumps({"redirect": response.url,
                                       "content": "Error, unknown content from redirect"})
            else:
                return Response(response.content)

        @app.route("/progress")
        def progress():
            url = request.args.get("url")
            if not url:
                return self.loge("No url given to check")
            return json.dumps("METHOD NOT IMPLEMENTED")
            # progress = self.get.progress(url)
            # if progress:
            #     return progress
            # else:
            #     return self.loge("No such url: {url}")

        @app.route("/update_links_cache")
        def update_links_cache():
            if not self.pdf_cache_helper:
                return self.loge("Cache helper is not available.")
            if not self.update_cache_run:
                self.update_cache_run = True
            if self.pdf_cache_helper.updating:
                return "Still updating cache from previous call"
            files = self.pdf_cache_helper.cache_needs_updating
            if files:
                self.pdf_cache_helper.update_cache()
                return self.logi(f"Updating cache for {len(files)} files")
            else:
                return self.logi("Nothing to update")

        @app.route("/force_stop_update_links_cache")
        def foce_stop_update_cache():
            if not self.pdf_cache_helper:
                return self.loge("Cache helper is not available.")
            if not self.update_cache_run:
                return self.logi("Update cache was never called")
            else:
                self.pdf_cache_helper.stop_update()
                return self.logi("Sent signal to stop updating cache")

        @app.route("/links_cache_updated")
        def cache_updated():
            if not self.pdf_cache_helper:
                return self.loge("Cache helper is not available.")
            if not self.update_cache_run:
                return self.logi("Update cache was never called.")
            elif self.pdf_cache_helper.updating:
                return self.logi("Still updating cache")
            elif self.pdf_cache_helper.finished:
                return self.logi("Updated cache for all files")
            elif self.pdf_cache_helper.finished_with_errors:
                return self.logi("Updated cache with errors.")
            else:
                return self.logi("Nothing was updated in last call to update cache")

        @app.route("/check_proxies")
        def check_proxies():
            return self.check_proxies()

        @app.route("/get_cvf_url", methods=["GET"])
        def get_cvf_url():
            """Get CVPR or ICCV PDF url.
            """
            if "title" not in request.args:
                return self.loge("Error. Title not in request")
            else:
                title = request.args["title"]
            if "venue" not in request.args:
                return self.loge("Error. Venue not in request")
            else:
                venue = request.args["venue"].lower()
            year = request.args.get("year")
            maybe_link = self.cvf_helper.get_pdf_link(title, venue, year)
            if maybe_link:
                return maybe_link
            else:
                return f"{title} not found for {venue} in {year}"

        @app.route("/echo", methods=["GET"])
        def echo():
            """Return a string representationn of request args or echo."""
            if request.args:
                return "\n".join(k + " : " + v for k, v in request.args.items())
            else:
                return "echo"

        @app.route("/version", methods=["GET"])
        def version():
            "Return version"
            return f"ref-man python server {__version__}"

        # TODO: rest of helpers should also support proxy
        # CHECK: Why are the interfaces to _dblp_helper and arxiv_helper different?
        #        Ideally there should be a single specification
        _proxy = self.everything_proxies if self.proxy_everything else None
        dblp_fetch, _dblp_helper = dblp_helper(_proxy, verbose=True)

        @app.route("/dblp", methods=["POST"])
        def dblp():
            """Fetch from DBLP"""
            return post_json_wrapper(request, dblp_fetch, _dblp_helper,
                                     self.batch_size, "DBLP", self.logger)

        @app.route("/get_yaml", methods=["POST"])
        def get_yaml():
            data = request.json
            return yaml.dump(data)

        @app.route("/shutdown")
        def shutdown():
            if self.pdf_cache_helper:
                self.logd("Shutting down cache helper.")
                self.pdf_cache_helper.shutdown()
            func = request.environ.get('werkzeug.server.shutdown')
            if func:
                func()
            elif hasattr(self, "proc"):
                self.logd("werkzeug.server.shutdown not available")
                self.logd("Trying psutil terminate")
                t = Thread(target=_shutdown)
                t.start()
            return self.logi("Shutting down")

        def _shutdown():
            time.sleep(.1)
            if self.proc:
                p = psutil.Process(self.proc.pid)
                p.terminate()

    def run(self):
        "Run the server"
        if self.debug:
            self.logd(f"Started Ref Man Service version {__version__} in debug mode")
            serving.run_simple(self.host, self.port, app, threaded=False)
            self.proc = None
        else:
            self.logd(f"Started Ref Man Service version {__version__}")
            self.proc = Process(target=serving.run_simple,
                                args=(self.host, self.port, app),
                                kwargs={"threaded": self.threaded})
            self.proc.start()
