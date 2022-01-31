from typing import List, Dict, Optional, Union
import os
import json
import requests
from subprocess import Popen, PIPE
import shlex
from pathlib import Path


Cache = Dict[str, Dict[str, str]]
assoc = [(x, i) for i, x in enumerate(["acl", "arxiv", "corpus", "doi"])]


class FilesCache:
    """A files based Cache for Semantic Scholar data.

    The cache is a Dictionary of type :code:`Cache` where they keys are one of
    `["acl", "arxiv", "corpus", "doi"]` and values are a dictionary of that id
    type and the associated ss_id.

    Each ss_id is stored as a file with the same
    name as the ss_id and contains the data for the entry in JSON format.

    Args:
        root: root directory where all the metadata and the
              files data will be kept

    """
    def __init__(self, root: Path):
        if not root.exists():
            raise FileExistsError(f"{root} doesn't exist")
        self._root = root
        self._cache: Cache = {}
        self._rev_cache: Dict[str, List[str]] = {}
        self._files: List[str] = [*filter(lambda x: not x.endswith("~") and not x == "metadata",
                                          os.listdir(self._root))]

    def load(self):
        """Load the Semantic Scholar metadata from the disk.

        The cache is indexed as a file in :code:`metadata` and the file data itself is
        named as the Semantic Scholar :code:`corpusId` for the paper. We load metadata on
        startup and fetch the rest as needed.

        Args:
            data_dir: Directory where the cache is located

        """
        with open(os.path.join(self._root, "metadata")) as f:
            _cache = [*filter(None, f.read().split("\n"))]
        self._cache = {"acl": {}, "doi": {}, "arxiv": {}, "corpus": {}}
        self._rev_cache = {}
        dups = False
        for _ in _cache:
            c = _.split(",")
            if c[-1] in self._rev_cache:
                dups = True
                self._rev_cache[c[-1]] = [x or y for x, y in zip(self._rev_cache[c[-1]], c[:-1])]
            else:
                self._rev_cache[c[-1]] = c[:-1]
            for key, ind in assoc:
                if c[ind]:
                    self._cache[key][c[ind]] = c[-1]
        print(f"Loaded SS cache {len(self._rev_cache)} entries and " +
              f"{sum(len(x) for x in self._cache.values())} keys.")
        if dups:
            print("There were duplicates. Writing new metadata")
            self._dump_metadata()

    def _dump_metadata(self):
        """Dump metadata to disk"""
        with open(self._root.joinpath("metadata"), "w") as f:
            f.write("\n".join([",".join([*v, k]) for k, v in self._rev_cache.items()]))
        print("Dumped metadata")


    def put(self, acl_id: str, data: Dict[str, str]):
        """Update entry, save paper data and Semantic Scholar cache to disk.

        We read and write data for individual papers instead of one big json
        object.

        Args:
            data: data for the paper
            acl_id: Optional ACL Id for the paper

        """
        with open(os.path.join(self._root, data["paperId"]), "w") as f:
            json.dump(data, f)
        c = [acl_id if acl_id else "",
             data["arxivId"] if data["arxivId"] else "",
             str(data["corpusId"]),
             data["doi"] if data["doi"] else "",
             data["paperId"]]
        for key, ind in assoc:
            if c[ind]:
                self._cache[key][c[ind]] = c[-1]
        existing = self._rev_cache.get(c[-1], None)
        if existing:
            self._rev_cache[c[-1]] = [x or y for x, y in zip(self._rev_cache[c[-1]], c[:-1])]
        else:
            self._rev_cache[c[-1]] = c[:-1]
            with open(os.path.join(self._root, "metadata"), "a") as f:
                f.write("\n" + ",".join([*self._rev_cache[c[-1]], c[-1]]))
            print("Updated metadata")

    def get(self, id_type: str, ID: str, force: bool) -> Union[str, bytes]:
        """Get semantic scholar paper details

        The Semantic Scholar cache is checked first and if it's a miss then the
        details are fetched from the server and stored in the cache.

        Args:
            id_type: type of the paper identifier one of
                     `['ss', 'doi', 'mag', 'arxiv', 'acl', 'pubmed', 'corpus']`
            ID: paper identifier
            force: Force fetch from Semantic Scholar server, ignoring cache

        """
        urls = {"ss": f"https://api.semanticscholar.org/v1/paper/{ID}",
                "doi": f"https://api.semanticscholar.org/v1/paper/{ID}",
                "mag": f"https://api.semanticscholar.org/v1/paper/MAG:{ID}",
                "arxiv": f"https://api.semanticscholar.org/v1/paper/arXiv:{ID}",
                "acl": f"https://api.semanticscholar.org/v1/paper/ACL:{ID}",
                "pubmed": f"https://api.semanticscholar.org/v1/paper/PMID:{ID}",
                "corpus": f"https://api.semanticscholar.org/v1/paper/CorpusID:{ID}"}
        if id_type not in urls:
            return json.dumps("INVALID ID TYPE")
        elif id_type == "ss":
            ssid: Optional[str] = ID
        elif id_type in {"doi", "acl", "arxiv", "corpus"}:
            ssid = self._cache[id_type].get(ID, None)
        if not ssid or force:
            fetch_from_disk = False
        else:
            fetch_from_disk = True
        return self.fetch(fetch_from_disk, ssid, urls, id_type, ID, force)

    def fetch(self, fetch_from_disk: bool, ssid: Optional[str],
              urls: Dict[str, str], id_type: str, ID: str, force: bool):
        """Subroutine to fetch from either disk or Semantic Scholar.

        Args:
            fetch_from_disk: Fetch from disk if True
            ssid: Optional Semantic Scholar ID
            urls: A dictionary of urls for each ID type
            id_type: type of the paper identifier one of
                     `['ss', 'doi', 'mag', 'arxiv', 'acl', 'pubmed', 'corpus']`
            ID: paper identifier
            force: Force fetch from Semantic Scholar server if True, ignoring cache

        """
        if fetch_from_disk and ssid:
            print(f"Fetching {ssid} from disk")
            data_file = self._root.joinpath(ssid)
            if data_file.exists():
                with open(data_file, "rb") as f:
                    data = f.read()
                return data
            else:
                print(f"File for {ssid} not present on disk. Will fetch.")
                url = f"https://api.semanticscholar.org/v1/paper/{ssid}"
                return self.fetch_from_ss(url)
        else:
            acl_id = ""
            if id_type == "acl":
                acl_id = ID
            url = urls[id_type] + "?include_unknown_references=true"
            if force and ssid:
                print(f"Forced Fetching for {ssid}")
            else:
                print(f"Data not in cache for {id_type}, {ID}. Fetching")
            return self.fetch_from_ss(url, acl_id)

    def fetch_from_ss(self, url, acl_id=""):
        """Fetch paper data from SS for url.

        Args:
            url: Full url of paper data

        """
        response = requests.get(url)
        if response.status_code == 200:
            self.put(acl_id, json.loads(response.content))
            return response.content  # already JSON
        else:
            print(f"Server error. Could not fetch")
            return json.dumps(None)



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
