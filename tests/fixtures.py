import os
import sys
import shutil
from threading import Thread
from pathlib import Path
sys.path.append('.')

import requests
import pytest

from ref_man_py.semantic_scholar import FilesCache
from ref_man_py.semantic_scholar import SemanticScholar
from ref_man_py.server import Server


@pytest.fixture
def cache_or_s2api(request):
    if hasattr(request, "param") and request.param == "cache":
        shutil.copy("tests/cache_data/metadata.bak",
                    "tests/cache_data/metadata")
        cache = FilesCache(Path("tests/cache_data/"))
    else:
        with open("tests/cache_data/metadata.bak") as f:
            temp = filter(None, f.read().split("\n"))
        temp = [t.rsplit(",", 1) for t in temp]
        for t in temp:
            t.insert(-1, ",,")
        with open("tests/cache_data/metadata", "w") as f:
            f.write("\n".join([",".join(t) for t in temp]))
        cache = SemanticScholar(cache_dir="tests/cache_data/")
    return cache


@pytest.fixture
def s2():
    with open("tests/cache_data/metadata.bak") as f:
        temp = filter(None, f.read().split("\n"))
    temp = [t.rsplit(",", 1) for t in temp]
    for t in temp:
        t.insert(-1, ",,")
    with open("tests/cache_data/metadata", "w") as f:
        f.write("\n".join([",".join(t) for t in temp]))
    return SemanticScholar(cache_dir="tests/cache_data/")


@pytest.fixture
def cache():
    shutil.copy("tests/cache_data/metadata.bak",
                "tests/cache_data/metadata")
    cache = FilesCache(Path("tests/cache_data/"))
    return cache


@pytest.fixture
def cache_files():
    cache_dir = "tests/cache_data"
    return [x for x in os.listdir(cache_dir) if "metadata" not in x]


@pytest.fixture(scope="module")
def server():
    kwargs = {"host": "localhost",
              "port": 9998,
              "proxy_port": None,
              "proxy_everything": False,
              "proxy_everything_port": None,
              "data_dir": "tests/cache_data",
              "local_pdfs_dir": None,
              "remote_pdfs_dir": None,
              "remote_links_cache": None,
              "batch_size": 16,
              "chrome_debugger_path": None,
              "verbosity": "debug",
              "threaded": True}
    server = Server(**kwargs)
    server.run()
    yield server
    requests.get("http://localhost:9998/shutdown")
