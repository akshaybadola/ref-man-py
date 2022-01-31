import pytest
from pathlib import Path
from ref_man.semantic_scholar import FilesCache


@pytest.fixture
def cache():
    cache = FilesCache(Path("tests/cache_data/"))
    return cache
