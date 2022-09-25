from typing import Dict, cast, Optional
import glob
import gzip
import json
from collections import defaultdict
import os
import pickle
from pathlib import Path

from common_pyutil.monitor import Timer


__doc__ = """Module to process Semantic Scholar Data."""
timer = Timer()


def parse_citations(root_dir):
    citations = defaultdict(set)
    filenames = glob.glob(root_dir + "*gz")
    for filename in filenames:
        with gzip.open(filename, "rt") as s2_file:
            for i, line in enumerate(s2_file):
                data = json.loads(line)
                if data["citedcorpusid"] and data["citingcorpusid"]:
                    a, b = int(data["citedcorpusid"]), int(data["citingcorpusid"])
                if not (i % 999999):
                    print(f"{i+1} done for file {filename}")
                citations[a].add(b)
    out_file = os.path.join(root_dir, "citations.pkl")
    print(f"Writing file {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(citations, f)


def save_temp(output_dir, data, i):
    """Dump a temp pickle file of adjacency list

    Args:
        output_dir: Output directory to save the file
        temp: The temporary output file
        i: The numeric suffix for the file

    """
    with timer:
        with open(output_dir.joinpath(f"temp_{i:010}.pkl"), "wb") as f:
            pickle.dump(data, f)
    print(f"Dumped for {i} in {timer.time} seconds")


def split_and_dump_citations(input_dir, output_dir, citations, max_key):
    """Split and dump the citations

    Args:
        input_dir: Input Directory
        output_dir: Output Directory
        citations: Citations loaded from pickle file
        max_key: Max value of all keys


    """
    j = 0
    while True:
        temp = {}
        a = j * 1000000
        b = (j+1) * 1000000
        if os.path.exists(input_dir.joinpath(f"temp_{b:010}.pkl")):
            print(f"skipping for {b:010}")
            continue
        with timer:
            for i in range(a, b):
                if i in citations:
                    temp[i] = citations[i].copy()
                if i > max_key:
                    save_temp(output_dir, temp, b)
                    return
        print(f"Done for {b} in {timer.time} seconds")
        save_temp(temp, b)
        j += 1


def split_citations(root_dir: Path):
    """Read the citations.pkl file and split them based on corpus_id

    Args:
        root_dir: Root directory where citations reside


    """
    with timer:
        with open(root_dir.joinpath("citations.pkl"), "rb") as f:
            citations = pickle.load(f)
    print(f"Loaded citations in {timer.time} seconds")
    keys = [*citations.keys()]
    max_key = max(keys)
    split_and_dump_citations(root_dir, root_dir, citations, max_key)


def convert_keys_from_numpy(cache):
    """Convert cache keys from :class:`numpy.int64` to :class:`int`

            Used once when keys were taken from numpy

            Args:
                cache: :class:`RefsCache`

            """
    for i, cf in enumerate(cache.files.values()):
        print(f"Opening {i+1} file")
        with open(cf, "rb") as fp:
            data = pickle.load(fp)
        out_data = defaultdict(set)
        for k, v in data.items():
            out_data[int(k)] = v
        with open(cf, "wb") as fp:
            pickle.dump(out_data, fp)
        print(f"Done {i+1} file")


class CitationCache:
    """A Semantic Scholar Citations cache.

    Consists of pickle files of :class:`dict` entries with keys as :code:`citedPaper`
    and values of :code:`citingPaper`

    The pickle files are stored such that :code:`corpusId` of a :code:`citingPaper`
    is smaller than :code:`temp_{suffix}` where :code:`suffix` is an integer

    Args:
        root_dir: Root directory where cache resides


    """
    def __init__(self, root_dir: Path):
        self._root_dir = root_dir
        _root_dir = str(root_dir).removesuffix("/") + "/"
        files = glob.glob(_root_dir + "*.pkl")
        files.sort()
        _files: Dict[int, str] = {int(f.replace(_root_dir, "").
                                      replace("temp_", "").
                                      replace(".pkl", "")): f
                                  for f in files}
        self.files = _files
        self._cache: Dict[int, set] = {}

    @property
    def cache(self):
        """Cache to avoid reading files multiple times

        It's dictionary of type Dict[corpusId, set(corpusId)]

        """
        return self._cache


    def get_file(self, ID: int):
        """Get the file corresponding to a corpusId

        Args:
            ID: corpusId of a paper


        """
        for i, f in enumerate(self.files):
            if ID < f:
                print(f"Looking in file {f}")
                with timer:
                    with open(self.files[f], "rb") as fp:
                        data = pickle.load(fp)
                print(f"Loaded file {self.files[f]} in {timer.time} seconds")
                return data

    def get_citations(self, ID: int) -> Optional[set]:
        """Get all the citing papers for a corpusId

        Args:
            ID: corpusId of a paper


        """
        print(f"Searching for {ID}")
        if ID in self.cache:
            print(f"Have data for {ID} in cache")
            return self.cache[ID]
        else:
            data = self.get_file(ID)
            self.cache[ID] = data[ID].copy()
            if data and ID in data:
                return data[ID]
            else:
                print(f"Could not find reference data for {ID}")
                return None
