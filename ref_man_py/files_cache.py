from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import os
import json
import requests
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
