from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import os
import re
import json
import requests
from subprocess import Popen, PIPE
import shlex
from pathlib import Path
import asyncio

import aiohttp

from .filters import (year_filter, author_filter, num_citing_filter,
                      num_influential_count_filter, venue_filter, title_filter)


Cache = Dict[str, Dict[str, str]]


class SemanticScholar:
    """A Semantic Scholar API client with a files based Cache.

    The cache is a Dictionary of type :code:`Cache` where they keys are one of
    `["acl", "arxiv", "corpus", "doi", "mag", "url"]` and values are a dictionary
    of that id type and the associated ss_id.

    Each ss_id is stored as a file with the same
    name as the ss_id and contains the data for the entry in JSON format.

    Args:
        root: root directory where all the metadata and the
              files data will be kept

    """

    @classmethod
    @property
    def filters(cls) -> Dict[str, Callable]:
        """Allowed filters on the entries.

        ["year", "author", "num_citing", "influential_count", "venue", "title"]

        """
        _filters: Dict[str, Callable] = {"year": year_filter,
                                         "author": author_filter,
                                         "num_citing": num_citing_filter,
                                         "influential_count": num_influential_count_filter,
                                         "venue": venue_filter,
                                         "title": title_filter}
        return _filters

    def __init__(self, config_file=None, cache_dir=None):
        self.load_config(config_file)
        self._api_key = self._config.get("api_key", None)
        self._root_url = "https://api.semanticscholar.org/graph/v1"
        self._cache_dir = Path(cache_dir or self._config.get("cache", None))
        if not self._cache_dir or not Path(self._cache_dir).exists():
            raise FileExistsError(f"{self._cache_dir} doesn't exist")
        self._in_memory: Dict[str, Dict] = {}
        self._cache: Cache = {}
        self._rev_cache: Dict[str, List[str]] = {}
        self._files: List[str] = [*filter(lambda x: not x.endswith("~") and "metadata" not in x,
                                          os.listdir(self._cache_dir))]
        self._id_keys = ['DOI', 'MAG', 'ARXIV', 'ACL', 'PUBMED', 'URL', 'CorpusId']
        self._id_keys.sort()
        self.load_metadata()

    def id_types(self, id_type: str):
        return "CorpusId" if id_type.lower() == "corpusid" else id_type.upper()

    def load_config(self, config_file: Union[Path, str]):
        """Load a given :code:`config_file` from disk.

        Args:
            config_file: The config file to load. The config file is in json

        """
        self._config_file = config_file
        self._config = self.default_config.copy()
        if config_file:
            with open(self._config_file) as f:
                config = json.load(f)
        else:
            config = {}
        for k in self._config:
            if k in config:
                if isinstance(self._config[k], dict):
                    self._config[k].update(config[k])  # type: ignore
                else:
                    self._config[k] = config[k]

    @property
    def default_config(self) -> Dict[str, Dict[str, Any]]:
        """Generate a default config in case config file is not on disk.

        """
        return {"api_key": None,
                "search": {"limit": 10,
                           "fields": ['authors', 'abstract', 'title',
                                      'venue', 'paperId', 'year',
                                      'url', 'citationCount',
                                      'influentialCitationCount',
                                      'externalIds']},
                # NOTE: narrowing to category isn't supported on the current API
                # "default_category": ""},
                "details": {"limit": 100,
                            "fields": ['authors', 'abstract', 'title',
                                       'venue', 'paperId', 'year',
                                       'url', 'citationCount',
                                       'influentialCitationCount',
                                       'externalIds']},
                "citations": {"limit": 1000,
                              "fields": ['authors', 'abstract', 'title',
                                         'venue', 'paperId', 'year',
                                         'contexts',
                                         'url', 'citationCount',
                                         'influentialCitationCount',
                                         'externalIds']},
                "references": {"limit": 1000,
                               "fields": ['authors', 'abstract', 'title',
                                          'venue', 'paperId', 'year',
                                          'contexts',
                                          'url', 'citationCount',
                                          'influentialCitationCount',
                                          'externalIds']},
                "author": {"limit": 100,
                           "fields": ["authorId", "name"]},
                "author_papers": {"limit": 1000,
                                  "fields": ['authors', 'abstract', 'title',
                                             'venue', 'paperId', 'year',
                                             'url', 'citationCount',
                                             'influentialCitationCount',
                                             'externalIds']}}

    def load_metadata(self):
        """Load the Semantic Scholar metadata from the disk.

        The cache is indexed as a file in :code:`metadata` and the file data itself is
        named as the Semantic Scholar :code:`corpusId` for the paper. We load metadata on
        startup and fetch the rest as needed.

        Args:
            data_dir: Directory where the cache is located

        """
        metadata_file = self._cache_dir.joinpath("metadata")
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = [*filter(None, f.read().split("\n"))]
        else:
            metadata = []
        entry_length = len(self._id_keys) + 1
        invalid_entries = [[i, x, len(x.split(","))]
                           for i, x in enumerate(metadata)
                           if len(x.split(",")) != entry_length]
        if invalid_entries:
            print(f"Invalid entries in metadata {os.path.join(self._cache_dir, 'metadata')}.\n" +
                  f"At lines: {','.join([str(x[0]) for x in invalid_entries])}")
            # import sys
            # sys.exit(1)
            for k, v, _ in invalid_entries:
                rest, paper_id = v.rsplit(",", 1)
                val = ",".join([rest, ",,", paper_id])
                metadata[k] = val
        self._cache = {k: {} for k in self._id_keys}
        self._rev_cache = {}
        dups = False
        for _ in metadata:
            c = _.split(",")
            if c[-1] in self._rev_cache:
                dups = True
                self._rev_cache[c[-1]] = [x or y for x, y in zip(self._rev_cache[c[-1]], c[:-1])]
            else:
                self._rev_cache[c[-1]] = c[:-1]
            for ind, key in enumerate(self._id_keys):
                if c[ind]:
                    self._cache[key][c[ind]] = c[-1]
        print(f"Loaded SS cache {len(self._rev_cache)} entries and " +
              f"{sum(len(x) for x in self._cache.values())} keys.")
        if dups:
            print("There were duplicates. Writing new metadata")
            self.dump_metadata()

    def dump_metadata(self):
        """Dump metadata to disk.

        """
        with open(self._cache_dir.joinpath("metadata"), "w") as f:
            f.write("\n".join([",".join([*v, k]) for k, v in self._rev_cache.items()]))
        print("Dumped metadata")

    def update_metadata(self, paper_id: str):
        """Update Metadata on the disk

        Args:
            paper_id: The S2 paper ID

        """
        with open(os.path.join(self._cache_dir, "metadata"), "a") as f:
            f.write("\n" + ",".join([*map(str, self._rev_cache[paper_id]), paper_id]))
        print("Updated metadata")

    def _get(self, url: str) -> requests.Response:
        """Synchronously get a URL with the API key if present.

        Args:
            url: URL

        """
        if self._api_key:
            response = requests.get(url, headers={"x-api-key": self._api_key})
        else:
            response = requests.get(url)
        return response

    def put_all(self, data: Dict[str, Dict[str, str]]):
        """Update paper details, references and citations on disk.

        We read and write data for individual papers instead of one big json
        object.

        The data is strored as a dictionary with keys :code:`["details", "references", "citations"]`

        Args:
            data: data for the paper

        """
        details = data["details"]
        paper_id = details["paperId"]
        fname = os.path.join(self._cache_dir, str(paper_id))
        with open(fname, "w") as f:
            json.dump(data, f)
        print(f"Wrote file {fname}")
        ext_ids = {self.id_types(k): v for k, v in details["externalIds"].items()}  # type: ignore
        other_ids = [ext_ids.get(k, "") for k in self._id_keys]  # type: ignore
        for ind, key in enumerate(self._id_keys):
            if other_ids[ind]:
                self._cache[key][other_ids[ind]] = paper_id
        existing = self._rev_cache.get(paper_id, None)
        if existing:
            self._rev_cache[paper_id] = [x or y for x, y in
                                         zip(self._rev_cache[paper_id], other_ids)]
        else:
            self._rev_cache[paper_id] = other_ids
            self.update_metadata(paper_id)
        self._in_memory[paper_id] = data

    def put_citations(self, data: Dict[str, Dict[str, str]]):
        """Update citations in disk cache for a given paper.

        The paper data is already assumed to be on the disk

        Args:
            data: data for the paper

        """
        paper_id = str(data["paperId"])
        citations = data["citations"]
        with open(os.path.join(self._cache_dir, str(paper_id)), "r+") as f:
            existing_data = json.load(f)
            existing_data["citations"]["offset"].update(citations["offset"])
            existing_data["citations"]["next"].update(citations["next"])
            existing_data["citations"]["data"].update(citations["data"])
            json.dump(existing_data, f)
        other_ids = [data["externalIds"].get(k, "") for k in self._id_keys]  # type: ignore
        for ind, key in enumerate(self._id_keys):
            if other_ids[ind]:
                self._cache[key][other_ids[ind]] = paper_id

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data before sending as json.

        For compatibility with data fetched with older API.

        Args:
            data: data for the paper

        """
        if "details" in data:
            data["details"]["references"] = [x["citedPaper"] for x in data["references"]["data"]]
            data["details"]["citations"] = [x["citingPaper"] for x in data["citations"]["data"]]
            return data["details"]
        else:
            return data

    def details_url(self, ID: str):
        """Return the paper url for a given `ID`

        Args:
            ID: paper identifier

        """
        fields = ",".join(self._config["details"]["fields"])
        return f"{self._root_url}/paper/{ID}?fields={fields}"

    def citations_url(self, ID: str, num: int = 0):
        """Return the citations url for a given `ID`

        Args:
            ID: paper identifier

        """
        fields = ",".join(self._config["citations"]["fields"])
        limit = num or self._config["citations"]["limit"]
        return f"{self._root_url}/paper/{ID}/citations?fields={fields}&limit={limit}"

    def references_url(self, ID: str, num: int = 0):
        """Return the references url for a given `ID`

        Args:
            ID: paper identifier

        """
        fields = ",".join(self._config["references"]["fields"])
        limit = num or self._config["references"]["limit"]
        return f"{self._root_url}/paper/{ID}/references?fields={fields}&limit={limit}"

    def author_url(self, ID: str) -> str:
        fields = ",".join(self._config["author"]["fields"])
        limit = self._config["author"]["limit"]
        return f"{self._root_url}/author/{ID}?fields={fields}&limit={limit}"

    def author_papers_url(self, ID: str) -> str:
        fields = ",".join(self._config["author_papers"]["fields"])
        limit = self._config["author_papers"]["limit"]
        return f"{self._root_url}/author/{ID}/papers?fields={fields}&limit={limit}"

    async def _author(self, ID: str) -> Dict:
        urls = [f(ID) for f in [self.author_url, self.author_papers_url]]
        if self._api_key:
            headers = {"x-api-key": self._api_key}
        else:
            headers = {}
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [self._aget(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
        return dict(zip(["author", "papers"], results))

    def get_author_papers(self, ID: str) -> Dict:
        result = asyncio.run(self._author(ID))
        return {"author": result["author"],
                "papers": result["papers"]["data"]}

    async def _aget(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Asynchronously get a url.

        Args:
            sesssion: An :class:`aiohttp.ClientSession` instance
            url: The url to fetch

        """
        resp = await session.request('GET', url=url)
        data = await resp.json()
        return data

    async def _paper(self, ID: str) -> Dict[str, Dict]:
        """Asynchronously fetch paper details, references and citations.

        Gather and return the data

        Args:
            ID: paper identifier

        """
        urls = [f(ID) for f in [self.details_url,  # type: ignore
                                self.references_url,
                                self.citations_url]]
        if self._api_key:
            headers = {"x-api-key": self._api_key}
        else:
            headers = {}
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [self._aget(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
        return dict(zip(["details", "references", "citations"], results))

    def store_details_and_get(self, ID: str, no_transform) -> Dict:
        """Get paper details asynchronously and store them.

        Fetch paper details, references and citations async.

        Store data in cache

        Args:
            ID: paper identifier

        """
        result = asyncio.run(self._paper(ID))
        self.put_all(result)
        return result if no_transform else self.transform(result)

    def fetch_from_cache_or_service(self, have_metadata: bool,
                                    ID: str, force: bool,
                                    no_transform: bool) -> Dict:
        """Subroutine to fetch from either disk or Semantic Scholar.

        Args:
            have_metadata: We already had the metadata
            ID: paper identifier
            force: Force fetch from Semantic Scholar server if True, ignoring cache

        """
        if have_metadata:
            print(f"Checking for cached data for {ID}")
            data = self._check_cache(ID)
            if not force:
                if data is not None:
                    return data if no_transform else self.transform(data)
                else:
                    print(f"Details for {ID} not present on disk. Will fetch.")
                    return self.store_details_and_get(ID, no_transform)
            else:
                print(f"Force fetching from Semantic Scholar for {ID}")
                return self.store_details_and_get(ID, no_transform)
        else:
            print(f"Fetching from Semantic Scholar for {ID}")
            return self.store_details_and_get(ID, no_transform)

    def get_details_for_id(self, id_type: str, ID: str, force: bool) -> Union[str, Dict]:
        """Get paper details from Semantic Scholar Graph API

        The on disk cache is checked first and if it's a miss then the
        details are fetched from the server and stored in the cache.

        `force` force fetches the data from the API and updates the cache
        on the disk also.

        Args:
            id_type: type of the paper identifier one of
                     `['ss', 'doi', 'mag', 'arxiv', 'acl', 'pubmed', 'corpus']`
            ID: paper identifier
            force: Force fetch from Semantic Scholar server, ignoring cache

        """
        ids = {"doi": f"DOI:{ID}",
               "mag": f"MAG:{ID}",
               "arxiv": f"ARXIV:{ID}",
               "acl": f"ACL:{ID}",
               "pubmed": f"PMID:{ID}",
               "url": f"URL:{ID}",
               "corpus": f"CorpusID:{ID}"}
        if id_type == "ss":
            ssid = ID
            have_metadata = ssid in self._rev_cache
        elif id_type not in ids:
            return "INVALID ID TYPE"
        else:
            itype = self.id_types(id_type)
            ssid = self._cache[itype].get(ID, "")
            have_metadata = bool(ssid)
        return self.apply_limits(self.fetch_from_cache_or_service(
            have_metadata, ssid or ids[id_type], force, False))

    # CHECK: Is this function here just because I have references and citations?
    #        For consistency?
    def details(self, ID: str, force: bool = False) -> Union[str, Dict]:
        """Get details for paper with SSID :code:`ID`

        Args:
            ID: SSID of the paper
            force: Whether to force fetch from service

        """
        return self.get_details_for_id("ss", ID, force)

    def apply_limits(self, data):
        """Apply count limits to S2 data citations and references

        Args:
            data: S2 Data

        Limits are defined in configuration

        """
        _data = data.copy()
        if "citations" in data:
            limit = self._config["citations"]["limit"]
            _data["citations"] = _data["citations"][:limit]
        if "references" in data:
            limit = self._config["references"]["limit"]
            _data["references"] = _data["references"][:limit]
        return _data

    # TODO: Do I need two details functions?
    #       See details above
    # def get_all_details(self, ID: str):
    #     """Get details for paper with SSID :code:`ID`

    #     Args:
    #         ID: SSID of the paper

    #     Difference between this and :meth:`details` is that this one calls
    #     :meth:`fetch_from_cache_or_service` directly.

    #     """
    #     ssid = ID
    #     have_metadata = ssid in self._rev_cache
    #     return self.fetch_from_cache_or_service(have_metadata, ssid, False, True)

    def _get_details_from_disk(self, ID: str):
        data_file = self._cache_dir.joinpath(ID)
        if data_file.exists():
            print(f"Data for {ID} is on disk")
            with open(data_file, "rb") as f:
                data = f.read()
            return data

    def _validate_fields(self, data: Dict[str, Dict]) -> bool:
        details_fields = self._config["details"]["fields"].copy()
        references_fields = self._config["references"]["fields"].copy()
        check_contexts = False
        if "contexts" in references_fields:
            references_fields.remove("contexts")
            check_contexts = True
        citations_fields = self._config["citations"]["fields"].copy()
        if "contexts" in citations_fields:
            citations_fields.remove("contexts")
        if all(x in data for x in ["details", "references", "citations"]):
            valid_details = all([f in data["details"] for f in details_fields])
            valid_refs = all([f in data["references"]["data"][0]["citedPaper"]
                              for f in references_fields])
            valid_cites = all([f in data["citations"]["data"][0]["citingPaper"]
                               for f in citations_fields])
            if check_contexts:
                valid_refs = valid_refs and "contexts" in data["references"]["data"][0]
                valid_cites = valid_cites and "contexts" in data["citations"]["data"][0]
            return valid_details and valid_refs and valid_cites
        else:
            return False

    def _check_cache(self, ID: str) -> Optional[Dict]:
        """Check cache and return data for ID if found.

        First the `in_memory` cache is checked and then the on disk cache.

        Args:
            ID: Paper ID

        """
        if ID not in self._in_memory:
            print(f"Data for {ID} not in memory")
            data = self._get_details_from_disk(ID)
            if data:
                _data = json.loads(data)
                if self._validate_fields(_data):
                    self._in_memory[ID] = _data
                else:
                    print(f"Stale data for {ID}")
                    return None
        else:
            print(f"Data for {ID} in memory")
        if ID in self._in_memory:
            return self._in_memory[ID]
        else:
            return None

    def _citations(self, ID: str, num: int = 0):
        """Get all citations for a paperId :code:`ID` from Semantic Scholar Graph API

        Citations are fetched from the API directly according to the configuration

        Args:
            ID: paper identifier
            num: (Optional) Number of citations to fetch

        """
        url = self.citations_url(ID, num)
        return self._get(url)

    def _references(self, ID: str, num: int = 0):
        """Get all references for a paperId :code:`ID` from Semantic Scholar Graph API

        Args:
            ID: paper identifier
            num: (Optional) Number of references to fetch

        """
        url = self.references_url(ID, num)
        return self._get(url)

    def citations(self, ID: str, offset: int = 0,
                  count: Optional[int] = None):
        """Fetch citations for a paper according to range.

        The paper details including initial citations are already assumed to be
        in cache.

        If none of :code:`beg`, :code:`end`, :code:`count` are given, then
        send default limit number of citations.

        """
        data: Dict = self._check_cache(ID)  # type: ignore
        limit = count or self._config["citations"]["limit"]
        if data is None:
            self.details(ID)
            data = self._check_cache(ID)  # type: ignore
        retval = data["citations"]["data"][offset+1:offset+limit+1]
        return [x["citingPaper"] for x in retval]

    def next_citations(self, ID: str, num: int = 0) -> Optional[Dict]:
        """Fetch next citations for a paper if any.

        The paper details including initial citations are already assumed to be
        in cache.

        Args:
            ID: The paper ID

        """
        limit = num or self._config["citations"]["limit"]
        data = self._check_cache(ID)
        if data is not None and "next" in data["citations"]:
            paper_id = data["details"]["paperId"]
            fields = ",".join(self._config["citations"]["fields"])
            offset = data["citations"]["next"]
            url = f"{self._root_url}/paper/{ID}/citations?fields={fields}&limit={limit}&offset={offset}"
            citations = json.loads(self._get(url).content)
            if set([x["citingPaper"]["paperId"] for x in citations["data"]]).intersection(
                    [x["citingPaper"]["paperId"] for x in data["citations"]["data"]]):
                pass
            else:
                data["citations"]["data"].extend(citations["data"])
                data["citations"]["next"] = citations["next"]
                data["citations"]["offset"] = citations["offset"]
            with open(os.path.join(self._cache_dir, str(paper_id)), "w") as f:
                json.dump(data, f)
            return citations
        else:
            return None

    def filter_subr(self, key: str, values: List[Dict], filters: Dict[str, Any],
                    num: int) -> List[Dict]:
        """Subroutine for filtering papers

        Each filter function is called with the arguments and the results are AND'ed.

        Args:
            key: One of "references" or "citations"
            values: The values (paper details) to filter
            filters: Filter names and kwargs

        """
        retvals = []
        for val in values:
            status = True
            for k, v in filters.items():
                if key in val:
                    try:
                        status = status and self.filters[k](val[key], **v)  # kwargs only
                    except Exception as e:
                        print(f"Can't apply filter {k} on {val}: {e}")
                        status = False
                else:
                    status = False
            if status:
                retvals.append(val)
            if num and len(retvals) == num:
                break
        return [x[key] for x in retvals]

    def filter_citations(self, ID: str, filters: Dict[str, Any], num: int = 0) -> List[Dict]:
        """Filter citations based on given filters.

        Filters are json like dictionaries which have the filter name and the
        kwargs to each filter function call.  The functions are called in turn
        and only if all of them return :code:`True` does the filter return `True`.

        We can also do arbitrary combinations of AND and OR but that's a bit much.

        Args:
            ID: Paper ID
            filters: filter names and arguments

        """
        citations = self._check_cache(ID)["citations"]["data"]  # type: ignore
        return self.filter_subr("citingPaper", citations, filters, num)

    def filter_references(self, ID: str, filters: Dict[str, Any], num: int = 0):
        """LIke :meth:`filter_citations` but for references


        """
        references = self._check_cache(ID)["references"]["data"]  # type: ignore
        return self.filter_subr("citedPaper", references, filters, num)

    def search(self, query: str) -> Union[str, bytes]:
        """Search for query string on Semantic Scholar with graph search API.

        Args:
            query: query to search

        """
        terms = "+".join(query.split(" "))
        fields = ",".join(self._config["search"]["fields"])
        limit = self._config["search"]["limit"]
        url = f"{self._root_url}/paper/search?query={terms}&fields={fields}&limit={limit}"
        response = self._get(url)
        if response.status_code == 200:
            return response.content
        else:
            return '"Unknown Error"'


class SemanticSearch:
    # Example params:
    #
    # {'queryString': '', 'page': 1, 'pageSize': 10, 'sort': 'relevance',
    #  'authors': [], 'coAuthors': [], 'venues': [], 'yearFilter': None,
    #  'requireViewablePdf': False, 'publicationTypes': [], 'externalContentTypes': [],
    #  'fieldsOfStudy': ['computer-science'], 'useFallbackRankerService': False,
    #  'useFallbackSearchCluster': False, 'hydrateWithDdb': True, 'includeTldrs': False,
    #  'performTitleMatch': True, 'includeBadges': False, 'tldrModelVersion': 'v2.0.0',
    #  'getQuerySuggestions': False}

    """Semantic Scholar Search Module

    Args:
        debugger_path: Optional path to a JS debugger file.
                       Used for getting the arguments from Semantic Scholar Search
                       API from a chrome debugger websocket.
    """
    def __init__(self, debugger_path: Optional[Path]):
        self.params_file = Path(__file__).parent.joinpath("ss_default.json")
        with open(self.params_file) as f:
            self.default_params = json.load(f)
        self.params = self.default_params.copy()
        if debugger_path and debugger_path.exists():
            self.update_params(debugger_path)

    def update_params(self, debugger_path: Path) -> None:
        """Update the parameters for Semantic Scholar Search if possible

        Args:
            debugger_path: Optional path to a JS debugger file.

        """
        if debugger_path.exists():
            check_flag = False
            try:
                import psutil
                for p in psutil.process_iter():
                    cmd = p.cmdline()
                    if cmd and ("google-chrome" in cmd[0] or "chromium" in cmd[0]):
                        check_flag = True
                        break
            except Exception as e:
                print(e)
            if check_flag and cmd:
                print(f"Trying to update Semantic Scholar Search params")
                p = Popen(shlex.split(f"node {debugger_path}"), stdout=PIPE, stderr=PIPE)
                out, err = p.communicate()
                if err:
                    print("Chromium running but error communicating with it. Can't update params")
                    print(err.decode())
                else:
                    try:
                        vals = json.loads(out)
                        if "request" in vals and 'postData' in vals['request']:
                            if isinstance(vals['request']['postData'], dict):
                                vals = vals['request']['postData']
                            elif isinstance(vals['request']['postData'], str):
                                vals = json.loads(vals['request']['postData'])
                            else:
                                raise Exception("Not sure what data was sent")
                        self.params = vals.copy()
                        new_params = set(vals.keys()) - set(self.default_params.keys())
                        if new_params:
                            print(f"New params in SS Search {new_params}")
                        not_params = set(self.default_params.keys()) - set(vals.keys())
                        if not_params:
                            print(f"Params not in SS Search {not_params}")
                        values_to_update = {'performTitleMatch': True,
                                            'includeBadges': False,
                                            'includeTldrs': False,
                                            'fieldsOfStudy': ['computer-science']}
                        for k, v in values_to_update.items():
                            if k in self.params:
                                self.params[k] = v
                            else:
                                print(f"Could not update param {k}")
                        self.params['queryString'] = ''
                        print(f"Updated params {self.params}")
                        if new_params or not_params:
                            with open(self.params_file, "w") as f:
                                json.dump(self.params, f)
                            print(f"Dumpted params to file {self.params_file}")
                    except Exception as e:
                        print(f"Error updating params {e}. Will use default params")
                        self.params = self.default_params.copy()
            else:
                print("Chromium with debug port not running. Can't update params")
        else:
            print(f"Debug script path not given. Using default params")

    def semantic_scholar_search(self, query: str, cs_only: bool = False, **kwargs) ->\
            Union[str, bytes]:
        """Perform a search on semantic scholar and return the results.

        The results are returned in JSON format.  By default the search is
        performed in Computer Science subjects

        Args:
            query: The string to query
            cs_only: Whether search only in Computer Science category
            kwargs: Additional arguments to set for the search

        :code:`publicationTypes` in :code:`kwargs` can be ["Conference", "JournalArticle"]
        :code:`yearFilter` has to be a :class:`dict` of type {"max": 1995, "min": 1990}

        """
        params = self.params.copy()
        params["queryString"] = query
        if 'yearFilter' in kwargs:
            yearFilter = kwargs['yearFilter']
            if yearFilter and not ("min" in yearFilter and "max" in yearFilter and
                                   yearFilter["max"] > yearFilter["min"]):
                print("Invalid Year Filter. Disabling.")
                yearFilter = None
            params['yearFilter'] = yearFilter
        if cs_only:
            params['fieldsOfStudy'] = ['computer-science']
        else:
            params['fieldsOfStudy'] = []
        for k, v in kwargs.items():
            if k in params and isinstance(v, type(params[k])):
                params[k] = v
        headers = {'User-agent': 'Mozilla/5.0', 'Origin': 'https://www.semanticscholar.org'}
        print("Sending request to semanticscholar search with query" +
              f": {query} and params {self.params}")
        response = requests.post("https://www.semanticscholar.org/api/1/search",
                                 headers=headers, json=params)
        if response.status_code == 200:
            results = json.loads(response.content)["results"]
            print(f"Got {len(results)} results for query: {query}")
            return response.content  # already json
        else:
            return json.dumps({"error": f"ERROR for {query}, {str(response.content)}"})
