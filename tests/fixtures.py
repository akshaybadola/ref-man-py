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


@pytest.fixture(scope="session")
def service():
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
