import os
import sys
import shutil
from threading import Thread
from pathlib import Path
import time
sys.path.append('.')


import requests
import pytest

from ref_man_py.files_cache import FilesCache
from ref_man_py.semantic_scholar import SemanticScholar
from ref_man_py.service import RefMan as Server


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
    s2 = SemanticScholar(cache_dir="tests/cache_data/", refs_cache_dir="tests/refs_cache")
    s2_key = os.environ.get("S2_API_KEY")
    if s2_key:
        s2._api_key = s2_key
    return s2


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


@pytest.fixture(scope="session")
def server():
    shutil.copy("tests/cache_data/metadata.bak",
                "tests/cache_data/metadata")
    kwargs = {"host": "localhost",
              "port": 9998,
              "proxy_port": None,
              "proxy_everything": False,
              "proxy_everything_port": None,
              "data_dir": "tests/cache_data",
              "local_pdfs_dir": None,
              "remote_pdfs_dir": None,
              "remote_links_cache": None,
              "config_dir": Path(__file__).parent.joinpath("config"),
              "batch_size": 16,
              "chrome_debugger_path": None,
              "debug": False,
              "verbosity": "debug",
              "threaded": True}
    server = Server(**kwargs)
    s2_key = os.environ.get("S2_API_KEY")
    if s2_key:
        server.s2._api_key = s2_key
    server.run()
    time.sleep(1)
    yield server
    requests.get("http://localhost:9998/shutdown")
