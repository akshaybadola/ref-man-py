import pytest
import os
import json
import random


def test_cache_init(cache):
    assert cache._cache is not None
    assert not cache._cache
    assert not cache._rev_cache


def test_cache_load(cache):
    cache.load()
    assert cache._cache
    assert cache._rev_cache
    assert len(cache._rev_cache) == 31


def test_cache_load_with_dups(cache):
    with open(os.path.join(cache._root, "metadata")) as f:
        metadata = f.read().split("\n")
    for _ in range(5):
        metadata.append(random.choice(metadata))
    assert len(metadata) == 36
    with open(os.path.join(cache._root, "metadata"), "w") as f:
        f.write("\n".join(metadata))
    cache.load()
    assert len(cache._rev_cache) == 31


def test_cache_get_existing(cache):
    cache.load()
    files = [x for x in os.listdir(cache._root) if "metadata" not in x]
    fl = random.choice(files)
    data = cache.get("ss", fl, False)
    assert isinstance(data, bytes)
    assert len(data) > 0
    val = json.loads(data)
    assert isinstance(val, dict)


def test_cache_get_in_metadata_not_on_disk(cache):
    cache.load()
    files = [x for x in os.listdir(cache._root) if "metadata" not in x]
    fl = random.choice(files)
    fpath = cache._root.joinpath(fl)
    os.remove(fpath)
    assert not fpath.exists()
    data = cache.get("ss", fl, False)
    assert fpath.exists()
    assert isinstance(data, bytes)
    assert len(data) > 0
    val = json.loads(data)
    assert isinstance(val, dict)
    assert "paperId" in val
    assert val["paperId"] in cache._rev_cache


def test_cache_get_not_in_metadata_and_disk(cache):
    with open(os.path.join(cache._root, "metadata")) as f:
        metadata = f.read().split("\n")
    line = random.choice(metadata)
    metadata.remove(line)
    assert len(metadata) == 30
    with open(os.path.join(cache._root, "metadata"), "w") as f:
        f.write("\n".join(metadata))
    cache.load()
    key = line.split(",")[-1]
    # doesn't exist
    assert key not in cache._rev_cache
    fpath = cache._root.joinpath(key)
    os.remove(fpath)
    assert not fpath.exists()
    data = cache.get("ss", key, False)
    assert fpath.exists()
    assert isinstance(data, bytes)
    assert len(data) > 0
    val = json.loads(data)
    assert isinstance(val, dict)
    assert "paperId" in val
    # should exist now, called put
    assert val["paperId"] in cache._rev_cache


def test_cache_get_fail(cache):
    cache.load()
    data = cache.get("arxiv", "sdfdfdfdf", False)
    assert json.loads(data) is None
