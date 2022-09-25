from typing import List, Dict, Optional, Union, Tuple, Any, Callable, cast
import os
import json
import math
import requests
from pathlib import Path
import asyncio
import sys
if sys.version_info.minor > 7:
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import aiohttp
from common_pyutil.monitor import Timer

from .filters import (year_filter, author_filter, num_citing_filter,
                      num_influential_count_filter, venue_filter, title_filter)
from .data import CitationCache


timer = Timer()


class SubConfigType(TypedDict):
    limit: int
    fields: List[str]


class ConfigType(TypedDict):
    api_key: Optional[str]
    search: SubConfigType
    details: SubConfigType
    citations: SubConfigType
    references: SubConfigType
    author: SubConfigType
    author_papers: SubConfigType


CitationType = List[Dict[str, Union[str, Dict[str, str]]]]


class CitationsType(TypedDict, total=False):
    next: int
    offset: int
    data: CitationType


class DetailsDataType(TypedDict):
    paperId: str
    title: str
    citationCount: int
    influentialCitationCount: int
    venue: str
    year: str
    authors: List[Dict]
    citations: CitationsType
    references: CitationsType
    externalIds: Dict[str, Union[int, str]]


class StoredDataType(TypedDict):
    details: DetailsDataType
    citations: CitationsType
    references: CitationsType


Cache = Dict[str, Dict[str, str]]


def get_corpus_id(data: Dict):
    if "externalIds" in data:
        cid = data["externalIds"]["CorpusId"]
    elif "citingPaper" in data:
        cid = data["citingPaper"]["externalIds"]["CorpusId"]
    return cid


def citations_corpus_ids(data: Dict) -> List[int]:
    return [int(x["citingPaper"]["externalIds"]["CorpusId"])
            for x in data["citations"]["data"]]



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

    @property
    def filters(self) -> Dict[str, Callable]:
        """Allowed filters on the entries.

        ["year", "author", "num_citing", "influential_count", "venue", "title"]

        """
        _filters: Dict[str, Callable] = {"year": year_filter,
                                         "author": author_filter,
                                         "num_citing": num_citing_filter,
                                         "citationcount": num_citing_filter,
                                         "influential_count": num_influential_count_filter,
                                         "influentialcitationcount": num_influential_count_filter,
                                         "venue": venue_filter,
                                         "title": title_filter}
        return _filters

    def __init__(self, config_file=None, cache_dir=None, refs_cache_dir=None):
        self.load_config(config_file)
        self._api_key = self._config.get("api_key", None)
        self._root_url = "https://api.semanticscholar.org/graph/v1"
        self._cache_dir = Path(cache_dir or self._config.get("cache", None))
        if not self._cache_dir or not Path(self._cache_dir).exists():
            raise FileExistsError(f"{self._cache_dir} doesn't exist")
        self._refs_cache_dir = refs_cache_dir
        self._in_memory: Dict[str, StoredDataType] = {}
        self._cache: Cache = {}
        self._rev_cache: Dict[str, List[str]] = {}
        self._files: List[str] = [*filter(lambda x: not x.endswith("~") and "metadata" not in x,
                                          os.listdir(self._cache_dir))]
        self._id_keys = ['DOI', 'MAG', 'ARXIV', 'ACL', 'PUBMED', 'URL', 'CorpusId']
        self._id_keys.sort()
        self._batch_size = 100
        self._tolerance = 2
        self.load_metadata()
        self.load_refs_cache()

    def load_refs_cache(self):
        if self._refs_cache_dir and Path(self._refs_cache_dir).exists():
            self._refs_cache = CitationCache(self._refs_cache_dir)
        else:
            self._refs_cache = None

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
                # NOTE: Ignoring the whole thing TypedDict is hard to use for specific use cases
                if isinstance(self._config[k], dict):  # type: ignore
                    self._config[k].update(config[k])  # type: ignore
                else:
                    self._config[k] = config[k]  # type: ignore

    @property
    def default_config(self) -> ConfigType:
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

    @property
    def headers(self) -> Dict[str, str]:
        if self._api_key:
            return {"x-api-key": self._api_key}
        else:
            return {}

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
        response = requests.get(url, headers=self.headers)
        return response

    def _dump(self, ID: str, data: StoredDataType):
        fname = os.path.join(self._cache_dir, str(ID))
        if len(data["citations"]["data"]) > data["details"]["citationCount"]:
            data["details"]["citationCount"] = len(data["citations"]["data"])
        with timer:
            with open(fname, "w") as f:
                json.dump(data, f)
        print(f"Wrote file {fname} in {timer.time} seconds")

    def _update_citations(self, data: CitationsType,
                          existing_data: CitationsType):
        # NOTE: Ignoring as the Dict isn't uniform and it raises error
        data_cite_ids = {x["citingPaper"]["paperId"] for x in data["data"]}  # type: ignore
        for x in existing_data["data"]:
            if x["citingPaper"]["paperId"] not in data_cite_ids:  # type: ignore
                data["data"].append(x)
                if "next" in data:
                    data["next"] += 1

    def put_all(self, data: StoredDataType):
        """Update paper details, references and citations on disk.

        We read and write data for individual papers instead of one big json
        object.

        The data is strored as a dictionary with keys :code:`["details", "references", "citations"]`

        Args:
            data: data for the paper

        """
        details = data["details"]
        paper_id = details["paperId"]
        # NOTE: In case force updated and already some citations exist on disk
        existing_data = self._check_cache(paper_id)
        if existing_data is not None:
            self._update_citations(data["citations"], existing_data["citations"])
        self._dump(paper_id, data)
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

    def transform(self, data: Union[StoredDataType, DetailsDataType]) -> DetailsDataType:
        """Transform data before sending as json.

        For compatibility with data fetched with older API.

        Args:
            data: data for the paper

        """
        if "details" in data:
            data["details"]["references"] = [x["citedPaper"] for x in data["references"]["data"]]  # type: ignore
            data["details"]["citations"] = [x["citingPaper"] for x in data["citations"]["data"]]  # type: ignore
            return data["details"]  # type: ignore
        else:
            return data         # type: ignore

    def details_url(self, ID: str) -> str:
        """Return the paper url for a given `ID`

        Args:
            ID: paper identifier

        """
        fields = ",".join(self._config["details"]["fields"])
        return f"{self._root_url}/paper/{ID}?fields={fields}"

    def citations_url(self, ID: str, num: int = 0, offset: Optional[int] = None) -> str:
        """Return the citations url for a given `ID`

        Args:
            ID: paper identifier

        """
        fields = ",".join(self._config["citations"]["fields"])
        limit = num or self._config["citations"]["limit"]
        url = f"{self._root_url}/paper/{ID}/citations?fields={fields}&limit={limit}"
        if offset is not None:
            return url + f"&offset={offset}"
        else:
            return url

    def references_url(self, ID: str, num: int = 0) -> str:
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
        results = await self._get_some_urls(urls)
        # async with aiohttp.ClientSession(headers=self.headers) as session:
        #     tasks = [self._aget(session, url) for url in urls]
        #     results = await asyncio.gather(*tasks)
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

    async def _get_some_urls(self, urls: List[str]) -> List:
        """Get some URLs asynchronously

        Args:
            urls: List of URLs

        URLs are fetched with :class:`aiohttp.ClientSession` with api_key included

        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [self._aget(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
        return results

    async def _paper(self, ID: str) -> StoredDataType:
        """Asynchronously fetch paper details, references and citations.

        Gather and return the data

        Args:
            ID: paper identifier

        """
        urls = [f(ID) for f in [self.details_url,  # type: ignore
                                self.references_url,
                                self.citations_url]]
        print("FETCHING paper with _get_some_urls")
        results = await self._get_some_urls(urls)
        # async with aiohttp.ClientSession(headers=self.headers) as session:
        #     tasks = [self._aget(session, url) for url in urls]
        #     results = await asyncio.gather(*tasks)
        # NOTE: mypy can't resolve zip of async gather
        data: StoredDataType = dict(zip(["details", "references", "citations"], results))  # type: ignore
        return data

    def store_details_and_get(self, ID: str, no_transform) ->\
            Union[StoredDataType, DetailsDataType]:
        """Get paper details asynchronously and store them.

        Fetch paper details, references and citations async.

        Store data in cache

        Args:
            ID: paper identifier

        """
        result = asyncio.run(self._paper(ID))
        self.put_all(result)
        if no_transform:
            return result
        else:
            return self.transform(result)

    def fetch_from_cache_or_service(self, have_metadata: bool,
                                    ID: str, force: bool,
                                    no_transform: bool)\
            -> Union[StoredDataType, DetailsDataType]:
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
                    # NOTE: StoredData and data are both Dict
                    return data if no_transform else self.transform(data)  # type: ignore
                else:
                    print(f"Details for {ID} not present on disk. Will fetch.")
                    return self.store_details_and_get(ID, no_transform)
            else:
                print(f"Force fetching from Semantic Scholar for {ID}")
                return self.store_details_and_get(ID, no_transform)
        else:
            print(f"Fetching from Semantic Scholar for {ID}")
            return self.store_details_and_get(ID, no_transform)

    # TODO: Don't we expect the other data to be in cache?
    def _corpus_id(self, id_type: str, ID: str) -> Union[str, int]:
        ids = {"doi": f"DOI:{ID}",
               "mag": f"MAG:{ID}",
               "arxiv": f"ARXIV:{ID}",
               "acl": f"ACL:{ID}",
               "pubmed": f"PMID:{ID}",
               "url": f"URL:{ID}"}
        if id_type == "ss":
            ssid = ID
            have_metadata = ssid in self._rev_cache
        elif id_type not in ids:
            return "INVALID ID TYPE"
        else:
            itype = self.id_types(id_type)
            ssid = self._cache[itype].get(ID, "")
            have_metadata = bool(ssid)
        data = self.fetch_from_cache_or_service(
            have_metadata, ssid or ids[id_type], False, False)
        return data['externalIds']['CorpusId']

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

    def _get_details_from_disk(self, ID: str) -> Optional[StoredDataType]:
        data_file = self._cache_dir.joinpath(ID)
        if data_file.exists():
            print(f"Data for {ID} is on disk")
            with open(data_file, "rb") as f:
                data = f.read()
            return json.loads(data)
        else:
            return None

    def _validate_fields(self, data: StoredDataType) -> bool:
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
            if data["references"]["data"]:
                valid_refs = all([f in data["references"]["data"][0]["citedPaper"]
                                  for f in references_fields])
            else:
                valid_refs = True
            if data["citations"]["data"]:
                valid_cites = all([f in data["citations"]["data"][0]["citingPaper"]
                                   for f in citations_fields])
            else:
                valid_cites = True
            if check_contexts:
                valid_refs = valid_refs and data["references"]["data"] and\
                    "contexts" in data["references"]["data"][0]
                valid_cites = valid_cites and data["citations"]["data"] and\
                    "contexts" in data["citations"]["data"][0]
            return valid_details and valid_refs and valid_cites
        else:
            return False

    def _check_cache(self, ID: str) -> Optional[StoredDataType]:
        """Check cache and return data for ID if found.

        First the `in_memory` cache is checked and then the on disk cache.

        Args:
            ID: Paper ID

        """
        if ID not in self._in_memory:
            print(f"Data for {ID} not in memory")
            data = self._get_details_from_disk(ID)
            if data:
                if self._validate_fields(data):
                    self._in_memory[ID] = data
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

    def citations(self, ID: str, offset: int = 0, count: Optional[int] = None):
        """Fetch citations for a paper according to range.

        If none of :code:`beg`, :code:`end`, :code:`count` are given, then
        send default limit number of citations.

        """
        data = self.fetch_from_cache_or_service(True, ID, False, True)
        data = cast(StoredDataType, data)
        limit = count or self._config["citations"]["limit"]
        cite_data = data["citations"]["data"]
        if offset:
            if offset + limit > len(cite_data):
                self.next_citations(ID, limit, offset)
                data = self._check_cache(ID)  # type: ignore
                cite_data = data["citations"]["data"]
            retval = cite_data[offset:offset+limit]
        else:
            retval = cite_data[:limit]
        return [x["citingPaper"] for x in retval]

    # TODO: Although this fetches the appropriate data based on citations on disk
    #       the offset and limit handling is tricky and is not correct right now.
    # TODO: What if the num_citations change between the time we fetched earlier and now?
    def next_citations(self, ID: str, limit: int = 0, offset: int = 0) -> Optional[Dict]:
        """Fetch next citations for a paper if any.

        The paper details including initial citations are already assumed to be
        in cache.

        Args:
            ID: The paper ID

        """
        limit = limit or self._config["citations"]["limit"]
        data = self._check_cache(ID)
        if data is None:
            return {"error": f"Data for {ID} not in cache"}
        elif data is not None and "next" not in data["citations"]:
            return None
        else:
            if offset+limit > 10000:
                corpus_id = get_corpus_id(data["details"])
                citations = self._build_citations_from_stored_data(
                    corpus_id,
                    citations_corpus_ids(data),
                    offset, limit)
            else:
                paper_id = data["details"]["paperId"]
                _next = data["citations"]["next"]
                if offset:
                    limit += offset - _next
                offset = _next
                url = self.citations_url(paper_id, limit, offset)
                citations = json.loads(self._get(url).content)
            self._update_citations(data["citations"], citations)
            self._dump(paper_id, data)
            return citations

    def filter_subr(self, key: str, values: CitationType, filters: Dict[str, Any],
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
                        # kwargs only
                        status = status and self.filters[k](val[key], **v)
                    except Exception as e:
                        print(f"Can't apply filter {k} on {val}: {e}")
                        status = False
                else:
                    status = False
            if status:
                retvals.append(val)
            if num and len(retvals) == num:
                break
        # NOTE: Gives error because x[key] evals to Union[str, Dict[str, str]]
        return [x[key] for x in retvals]

    # TODO: This should either use `next_citations` or `next_citations` should use this
    async def _ensure_all_citations(self, ID: str) -> CitationsType:
        data = self._check_cache(ID)
        if data is not None:
            num_cites = data["details"]["citationCount"]
            offset = data["citations"].get("next", None)
            if offset:
                fields = ",".join(self._config["citations"]["fields"])
                iters = math.ceil((num_cites - offset) / 1000)
                urls = [f"{self._root_url}/paper/{ID}/citations?fields={fields}&limit=1000"
                        + f"&offset={offset + (i*1000)}"
                        for i in range(iters)]
                print(f"Will fetch {len(urls)} requests for citations")
                with timer:
                    results = await self._get_some_urls(urls)
                result: CitationsType = {"next": 0, "offset": 0, "data": []}
                cite_list = []
                errors = 0
                for x in results:
                    if "error" not in x:
                        cite_list.extend(x["data"])
                    else:
                        errors += 1
                result["data"] = cite_list
                if errors:
                    print(f"{errors} errors occured while fetching all citations for {ID}")
                offset = max(x["offset"] for x in results if "error" not in x)
                if all("next" in x for x in results):
                    result["next"] = max(x["next"] for x in results if "error" not in x)
                else:
                    result.pop("next")
                return result
            else:
                return data["citations"]
        else:
            msg = f"Paper data for {ID} should already exist"
            raise ValueError(msg)

    # TODO: Need to add condition such that if num_citations > 10000, then this
    #       function is called. And also perhaps, fetch first 1000 citations and
    #       update the stored data (if they're sorted by time)
    def _build_citations_from_stored_data(self,
                                          corpus_id: Union[int, str],
                                          existing_ids: Optional[List[int]] = None,
                                          offset: int = 0,
                                          limit: int = 0) -> CitationsType:
        """Build the citations data for a paper entry from cached data

        Args:
            corpus_id: Semantic Scholar CorpusId

        """

        refs_ids = self._refs_cache.get_refs(int(corpus_id))
        if not refs_ids:
            raise AttributeError(f"Not found for {corpus_id}")
        if existing_ids:
            need_ids = list(refs_ids - set(existing_ids))
        else:
            need_ids = list(refs_ids)
        if not limit:
            limit = len(need_ids)
        need_ids = need_ids[offset:offset+limit]
        # Remove contexts as that's not available in paper details
        fields = ",".join(self._config["citations"]["fields"]).replace(",contexts", "")
        urls = [f"{self._root_url}/paper/CorpusID:{ID}?fields={fields}"
                for ID in need_ids]
        citations = {"offset": 0, "data": []}
        batch_size = self._batch_size
        j = 0
        # Fetch batch_size examples at a time to prevent overloading the service
        _urls = urls[j*batch_size:(j+1)*batch_size]
        while _urls:
            print(f"Fetching for j {j} and {len(_urls)} urls")
            with timer:
                results = asyncio.run(self._get_some_urls(_urls))
            citations["data"].extend(results)  # type: ignore
            j += 1
            _urls = urls[j*batch_size:(j+1)*batch_size]
        citations["data"] = [{"citingPaper": x, "contexts": []}
                             for x in citations["data"]]  # type: ignore
        return citations                                  # type: ignore

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
        existing_data = self._check_cache(ID)
        if existing_data is None:
            msg = f"data should not be None for ID {ID}"
            return msg          # type: ignore
        else:
            if abs(existing_data["details"]["citationCount"] -
                   len(existing_data["citations"]["data"])) > self._tolerance:
                with timer:
                    data = asyncio.run(self._ensure_all_citations(ID))
                print(f"Fetched {len(data['data'])} in {timer.time} seconds")
                self._update_citations(data, existing_data["citations"])
                existing_data["citations"] = data
                self._dump(ID, existing_data)
            return self.filter_subr("citingPaper", existing_data["citations"]["data"], filters, num)

    def filter_references(self, ID: str, filters: Dict[str, Any], num: int = 0):
        """Like :meth:`filter_citations` but for references


        """
        data = self._check_cache(ID)
        if data is not None:
            references = data["references"]["data"]
        else:
            raise ValueError(f"Data for ID {ID} should be present")
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
            return json.dumps({"error": json.loads(response.content)})
