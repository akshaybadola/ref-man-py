from typing import List, Dict, Optional, Tuple, Union
import json
import operator
import os
import re
import time
from pathlib import Path
from threading import Thread
from multiprocessing import Process

import yaml
import requests
import psutil
from flask import Flask, request, Response
from werkzeug import serving

import bs4
from bs4 import BeautifulSoup

from common_pyutil.log import get_stream_logger

from . import __version__
from .const import default_headers
from .util import (fetch_url_info, fetch_url_info_parallel,
                   parallel_fetch, post_json_wrapper, check_proxy_port)
from .arxiv import arxiv_get, arxiv_fetch, arxiv_helper
from .dblp import dblp_helper
from .semantic_scholar import SemanticScholar
from .semantic_search import SemanticSearch
from .cache import CacheHelper


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
        refs_cache_dir: Directory of Semantic Scholar References Cache
        local_pdfs_dir: Local directory where the pdf files are stored.
        remote_pdfs_dir: Remote directory where the pdf files are stored.
        remote_links_cache: File mapping local pdfs to remote links.
        batch_size: Number of parallel requests to send in case parallel requests is
                    implemented for that method.
        chrome_debugger_path: Path for the chrome debugger script.
                              Used to validate the Semantic Scholar search api, as
                              the params can change sometimes. If it's not given then
                              default params are used and the user must update the params
                              in case of an error.
        debug: Whether to start the service in debug mode
        verbosity: Verbosity control
        threaded: Start the flask server in threaded mode. Defaults to :code:`True`.

    :code:`remote_pdfs_dir` has to be an :code:`rclone` path and the pdf files from
    :code:`local_pdfs_dir` is synced to that with :code:`rclone`.

    """
    # TODO: Generate default config and place in config dir.
    #       As of now only SemanticScholar configuration is loaded if it exists
    # TODO: Also allow the user to add config options like request headers,
    #       proxy ports, data_dir etc.
    def __init__(self, host: str, port: int, proxy_port: int, proxy_everything: bool,
                 proxy_everything_port: int, data_dir: Path, refs_cache_dir: Path,
                 config_dir: Path, local_pdfs_dir: Path,
                 remote_pdfs_dir: Path, remote_links_cache: Path,
                 batch_size: int, chrome_debugger_path: Path,
                 debug: bool, verbosity: str, threaded: bool):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.refs_cache_dir = refs_cache_dir
        self.local_pdfs_dir = local_pdfs_dir
        self.remote_pdfs_dir = remote_pdfs_dir
        self.remote_links_cache = remote_links_cache
        self.proxy_port = proxy_port
        self.proxy_everything = proxy_everything
        self.proxy_everything_port = proxy_everything_port
        self.chrome_debugger_path = Path(chrome_debugger_path) if chrome_debugger_path else None
        if not self.config_dir.exists():
            os.makedirs(self.config_dir)
        self.config_file: Optional[Path] = self.config_dir.joinpath("config.json")
        self.config_file = self.config_file if self.config_file.exists() else None
        self.debug = debug
        self.verbosity = verbosity
        self.threaded = threaded

        self._requests_timeout = 5
        self._requests_fetch_url_timeout = 60

        self.init_loggers()
        if self.config_file:
            self.logi(f"Loaded config file {self.config_file}")

        self.cvf_url_root = "https://openaccess.thecvf.com"
        self.soups: Dict[Tuple, bs4.Tag] = {}
        self.cvf_pdf_links: Dict[Tuple, List[str]] = {}
        # self.load_cvf_files()
        self.load_cvf_pdf_links()
        self.s2 = SemanticScholar(config_file=self.config_file, cache_dir=self.data_dir,
                                  refs_cache_dir=self.refs_cache_dir)
        self.init_remote_cache()

        # NOTE: Checks only once for the proxy, see util.check_proxy
        #       for a persistent solution
        self.check_proxies()
        # NOTE: Although this is still use here, SS may phase it out
        #       in favour of the graph/search
        self.semantic_search = SemanticSearch(self.chrome_debugger_path)
        self._app = app
        self.init_routes()

    def _get(self, url: str, **kwargs) -> requests.Response:
        return requests.get(url, timeout=self._requests_timeout, **kwargs)

    def _get_url(self, url: str, **kwargs) -> requests.Response:
        return requests.get(url, timeout=self._requests_fetch_url_timeout, **kwargs)

    def init_remote_cache(self):
        """Initialize cache (and map) of remote and local pdf files.

        The files can be synced from and to an :code:`rclone` remote.

        """
        self.update_cache_run = None
        if self.local_pdfs_dir and self.remote_pdfs_dir and self.remote_links_cache:
            self.pdf_cache_helper: Optional[CacheHelper] =\
                CacheHelper(self.local_pdfs_dir, Path(self.remote_pdfs_dir),
                            Path(self.remote_links_cache), self.logger)
        else:
            self.pdf_cache_helper = None
            self.logger.warning("All arguments required for pdf cache not given.\n"
                                "Will not maintain remote pdf links cache.")

    def init_loggers(self):
        # We set "error" to warning
        verbosity_levels = {"info", "error", "debug"}
        if self.verbosity not in verbosity_levels:
            self.verbosity = "info"
            self.logger = get_stream_logger("ref_man_logger", log_level=self.verbosity)
            self.logger.warning(f"{self.verbosity} was not in known levels. " +
                                f"Set to {self.verbosity}")
        else:
            self.logger = get_stream_logger("ref_man_logger", log_level=self.verbosity)
            self.logger.debug(f"Log level is set to {self.verbosity}.")

    def load_cvf_files(self):
        """Load the CVF Soups from HTML files.

        XML parses via :class:`BeautifulSoup` are maintained for easy
        fetching of an article in case it's availble.

        """
        self.cvf_files = [os.path.join(self.config_dir, f)
                          for f in os.listdir(self.config_dir)
                          if re.match(r'^(cvpr|iccv)', f.lower())
                          and not f.endswith("_pdfs")]
        self.logger.debug("Loading CVF soups.")
        for cvf in self.cvf_files:
            if not cvf.endswith("_pdfs"):
                match = re.match(r'^(cvpr|iccv)(.*?)([0-9]+)',
                                 Path(cvf).name, flags=re.IGNORECASE)
                if match:
                    venue, _, year = map(str.lower, match.groups())
                    with open(cvf) as f:
                        self.soups[(venue, year)] = BeautifulSoup(f.read(), features="lxml")
                else:
                    self.logger.error(f"Could not load file {cvf}")
        self.logger.debug(f"Loaded conference files {self.soups.keys()}")

    def load_cvf_pdf_links(self):
        """Load the CVF PDF links saved from HTML files.

        """
        self.cvf_pdf_link_files = [os.path.join(self.config_dir, f)
                                   for f in os.listdir(self.config_dir)
                                   if re.match(r'^(cvpr|iccv)', f.lower())
                                   and f.endswith("_pdfs")]
        self.logger.debug("Loading CVF pdf links.")
        for fname in self.cvf_pdf_link_files:
            match = re.match(r'^(cvpr|iccv)(.*?)([0-9]+)',
                             Path(fname).name, flags=re.IGNORECASE)
            if match:
                venue, _, year = map(str.lower, match.groups())
                with open(fname) as f:
                    self.cvf_pdf_links[(venue, year)] = f.read().split("\n")
            else:
                self.logger.error(f"Could not load pdf links from {fname}")
        self.logger.debug(f"Loaded PDF links {self.cvf_pdf_links.keys()}")

    def maybe_download_cvf_day_pages(self, response, venue, year):
        soup = BeautifulSoup(response.content, features="lxml")
        links = soup.find_all("a")
        regexp = f"(/)?({venue.upper()}{year}(.py)?).+"
        last_link_attrs = links[-1].attrs
        venue_match = re.match(regexp, last_link_attrs['href'])
        venue_group = venue_match and venue_match.group(2)
        if "href" in last_link_attrs and venue_group:
            day_links = [*filter(lambda x: re.match(r"Day [0-9]+?: ([0-9-+])", x.text),
                                 soup.find_all("a"))]
            content = []
            for i, dl in enumerate(day_links):
                day = re.match(r"Day [0-9]+?: ([0-9-]+)", dl.text).groups()[0]
                d_url = f"{self.cvf_url_root}/{venue_group}?day={day}"
                resp = self._get(d_url)
                if not resp.ok:
                    err = f"Status code {response.status_code} for {d_url}"
                    raise requests.HTTPError(err)
                content.append(resp.content)
                self.logd(f"Fetched page {i+1} for {venue.upper()}{year} and {day}")
            soup_content = BeautifulSoup("")
            for c in content:
                soup_content.extend(BeautifulSoup(c, features="lxml").html)
            return soup_content.decode()
        self.logd(f"Fetched page for {venue.upper()}{year}")
        return response.content.decode()

    def download_cvf_page_and_update_soups(self, venue, year):
        url = f"{self.cvf_url_root}/{venue.upper()}{year}"
        response = self._get(url)
        if response.ok:
            content = self.maybe_download_cvf_day_pages(response, venue, year)
        else:
            err = f"Status code {response.status_code} for {url}"
            raise requests.HTTPError(err)
        fname = self.config_dir.joinpath(f"{venue.upper()}{year}")
        with open(fname, "w") as f:
            f.write(content)
        with open(fname) as f:
            self.soups[(venue.lower(), year)] = BeautifulSoup(content, features="lxml")

    def return_link_subr(self, title, matches) -> str:
        if not matches:
            return f"URL Not found for {title}"
        elif len(matches) == 1:
            href = matches[0].group(0)
        else:
            matches.sort(key=lambda x: operator.abs(operator.sub(*x.span())))
            href = matches[-1].group(0)
        href = "https://openaccess.thecvf.com/" + href.lstrip("/")
        return f"{title};{href}"

    def find_link_from_soups(self, keys, title) -> Optional[str]:
        links = []
        for k in keys:
            links.extend(self.soups[k].find_all("a"))
        if links:
            regexp = ".*" + ".*".join([*filter(None, title.split(" "))][:3]) + ".*\\.pdf$"
            matches = [*filter(None, map(lambda x: re.match(regexp, x["href"], flags=re.IGNORECASE)
                                         if "href" in x.attrs else None, links))]
            return self.return_link_subr(title, matches)
        else:
            return None

    def find_pdf_link(self, keys, title) -> Optional[str]:
        links = []
        for k in keys:
            links.extend(self.cvf_pdf_links[k])
        if links:
            regexp = ".*" + ".*".join([*filter(None, title.split(" "))][:3]) + ".*\\.pdf$"
            matches = [*filter(None, map(lambda x: re.match(regexp, x, flags=re.IGNORECASE), links))]
            return self.return_link_subr(title, matches)
        else:
            return None

    def save_cvf_pdf_links_and_update(self, venue: str, year: str, soup) -> None:
        links = soup.find_all("a")
        pdf_links = [x["href"] for x in links
                     if "href" in x.attrs and x["href"].endswith(".pdf")]
        fname = self.config_dir.joinpath(f"{venue.upper()}{year}_pdfs")
        with open(fname, "w") as f:
            f.write("\n".join(pdf_links))
        self.cvf_pdf_links[(venue, year)] = pdf_links

    def read_cvf_pdf_links(self, venue: str, year: str) -> List[str]:
        fname = self.config_dir.joinpath(f"{venue.upper()}{year}_pdfs")
        with open(fname) as f:
            pdf_links = f.read().split("\n")
        return pdf_links

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
        msgs: List[str] = []
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
        def s2_paper():
            if request.method == "GET":
                if "id" in request.args:
                    id = request.args["id"]
                else:
                    return json.dumps("NO ID GIVEN")
                if "id_type" in request.args:
                    id_type = request.args["id_type"]
                else:
                    return json.dumps("NO ID_TYPE GIVEN")
                force = True if "force" in request.args else False
                all_data = True if "all" in request.args else False
                data = self.s2.get_details_for_id(id_type, id, force, all_data)
                return json.dumps(data)
            else:
                return json.dumps("METHOD NOT IMPLEMENTED")

        @app.route("/s2_corpus_id", methods=["GET"])
        def s2_corpus_id():
            if "id" in request.args:
                id = request.args["id"]
            else:
                return json.dumps("NO ID GIVEN")
            if "id_type" in request.args:
                id_type = request.args["id_type"]
            else:
                return json.dumps("NO ID_TYPE GIVEN")
            data = self.s2.get_corpus_id(id_type, id)
            return json.dumps(data)

        @app.route("/s2_config", methods=["GET"])
        def s2_config():
            return json.dumps(self.s2._config)

        @app.route("/s2_details/<ssid>", methods=["GET"])
        def s2_details(ssid: str) -> Union[str, bytes]:
            if "force" in request.args:
                force = True
            else:
                force = False
            return json.dumps(self.s2.details(ssid, force))

        @app.route("/s2_all_details/<ssid>", methods=["GET", "POST"])
        def s2_all_details(ssid: str) -> Union[str, bytes]:
            """Get paper, metadata, references and ALL citations for SSID."""
            if "force" in request.args:
                print("Force not supported with s2_all_details")
            if request.method == "GET":
                return json.dumps(self.s2.details(ssid, all_data=True))
            else:
                return json.dumps("METHOD NOT IMPLEMENTED")

        def s2_citations_references_subr(request, ssid: str, key) -> Union[str, bytes]:
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
            if request.method == "GET":
                if "filters" in request.args:
                    return json.dumps("FILTERS NOT SUPPORTED WITH GET")
                values = func(ssid, offset=offset, count=count)  # type: ignore
                return json.dumps(values)
            else:
                if self.debug and hasattr(request, "json"):
                    print("REQUEST JSON", request.json)
                data = request.json
                if not data or (data and "filters" not in data):
                    return json.dumps("METHOD NOT IMPLEMENTED IF filters NOT GIVEN")
                else:
                    filters = data["filters"]
                    func = self.s2.filter_citations if key == "citations" else\
                        self.s2.filter_references
                    return json.dumps(func(ssid, filters, count))

        @app.route("/s2_citations/<ssid>", methods=["GET", "POST"])
        def s2_citations(ssid: str) -> Union[str, bytes]:
            """Get the citations for a paper from S2 graph api.

            Requires an :code:`ssid` for the paper. Optional :code:`count` and
            :code:`filters` can be given in the request as arguments.

            See :meth:`s2_citations_references_subr` for details

            """
            return s2_citations_references_subr(request, ssid, "citations")

        @app.route("/s2_references/<ssid>", methods=["GET", "POST"])
        def s2_references(ssid: str) -> Union[str, bytes]:
            """Get the references for a paper from S2 graph api.

            Requires an :code:`ssid` for the paper. Optional :code:`count` and
            :code:`filters` can be given in the request as arguments.

            See :meth:`s2_citations_references_subr` for details

            """
            return s2_citations_references_subr(request, ssid, "references")

        @app.route("/s2_next_citations/<ssid>", methods=["GET", "POST"])
        def s2_next_citations(ssid: str) -> Union[str, bytes]:
            """Get the next citations from S2 graph api.

            See :meth:`SemanticScholar.next_citations` for implementation details.
            """
            if request.method == "GET":
                if "filters" in request.args:
                    return json.dumps("FILTERS NOT SUPPORTED WITH GET")
                if "count" in request.args:
                    count = int(request.args["count"])
                    if count > 10000:
                        json.dumps("MAX 10000 CITATIONS CAN BE FETCHED AT ONCE.")
                else:
                    count = 0
                return json.dumps(self.s2.next_citations(ssid, count))
            else:
                # TODO: Not Implemented yet
                return json.dumps("METHOD NOT IMPLEMENTED")

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

        @app.route("/recommendations", methods=["GET", "POST"])
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
                pos_ids = data.get("pos-ids", None)
                neg_ids = data.get("neg-ids", None)
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
                return self.s2.search(query)
            else:
                return json.dumps("METHOD NOT IMPLEMENTED")

        @app.route("/semantic_scholar_search", methods=["GET", "POST"])
        def ss_search():
            """Search Semantic Scholar for a query string.

            This is different than graph api requests."""
            if "q" in request.args and request.args["q"]:
                query = request.args["q"]
            else:
                return json.dumps("NO QUERY GIVEN or EMPTY QUERY")
            if request.method == "GET":
                return self.semantic_search.semantic_scholar_search(query)
            else:
                if request.json:
                    kwargs = request.json
                else:
                    kwargs = {}
                return self.semantic_search.semantic_scholar_search(query, **kwargs)

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
                if request.args.get("proxy_port"):
                    self.proxy_port = request.args.get("proxy_port")
                if request.args.get("proxy_everything_port"):
                    self.proxy_everything_port = request.args.get("proxy_everything_port")
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
                    response = self._get_url(url, headers=default_headers, proxies=self.proxies)
                except requests.exceptions.Timeout:
                    self.logger.error("Proxy not reachable. Fetching without proxy")
                    self.proxies = None
                    response = self._get_url(url, headers=default_headers)
                except requests.exceptions.ProxyError:
                    self.logger.error("Proxy not reachable. Fetching without proxy")
                    self.proxies = None
                    response = self._get_url(url, headers=default_headers)
            else:
                if not noproxy:
                    self.logger.warning("Proxy dead. Fetching without proxy")
                response = self._get_url(url, headers=default_headers)
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

        @app.route("/force_stop_update_cache")
        def foce_stop_update_cache():
            if not self.pdf_cache_helper:
                return self.loge("Cache helper is not available.")
            if not self.update_cache_run:
                return self.logi("Update cache was never called")
            else:
                self.pdf_cache_helper.stop_update()
                return self.logi("Sent signal to stop updating cache")

        @app.route("/cache_updated")
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
            if year:
                keys = [(v, y) for v, y in self.cvf_pdf_links if v == venue and y == year]
            else:
                keys = [(v, y) for v, y in self.cvf_pdf_links if v == venue]
                year = ",".join([y for v, y in self.soups if v == venue])
            if not keys and year:
                self.logd(f"Fetching page(s) for {venue.upper()}{year}")
                self.download_cvf_page_and_update_soups(venue, year)
                self.save_cvf_pdf_links_and_update(venue, year, self.soups[(venue, year)])
                keys = [(v, y) for v, y in self.soups if v == venue and y == year]
            # maybe_link = self.find_soup(keys, title)
            maybe_link = self.find_pdf_link(keys, title)
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
            serving.run_simple(self.host, self.port, app, self.threaded)
            self.proc = None
        else:
            self.logd(f"Started Ref Man Service version {__version__}")
            self.proc = Process(target=serving.run_simple,
                                args=(self.host, self.port, app),
                                kwargs={"threaded": self.threaded})
            self.proc.start()
