import pytest
import os
import json
import random
from ref_man_py.semantic_scholar import SemanticScholar


def test_s2_init(s2):
    assert s2._cache is not None
    assert s2._cache is not None
    assert s2._rev_cache is not None
    assert len(s2._rev_cache) == 31


def test_s2_load_cache_with_dups(s2):
    with open(os.path.join(s2._cache_dir, "metadata")) as f:
        metadata = f.read().split("\n")
    for _ in range(5):
        metadata.append(random.choice(metadata))
    assert len(metadata) == 36
    with open(os.path.join(s2._cache_dir, "metadata"), "w") as f:
        f.write("\n".join(metadata))
    api = SemanticScholar(cache_dir="tests/cache_data/")
    assert len(api._rev_cache) == 31


def test_s2_cache_get_details_on_disk(s2, cache_files):
    files = [x for x in os.listdir(s2._cache_dir) if "metadata" not in x]
    fl = random.choice(files)
    data = s2.get_details_for_id("ss", fl, False)
    assert isinstance(data, dict)
    assert len(data) > 0
    assert "paperId" in data


def test_s2_cache_get_in_metadata_not_on_disk(s2, cache_files):
    files = cache_files
    fl = random.choice(files)
    fpath = s2._cache_dir.joinpath(fl)
    os.remove(fpath)
    assert not fpath.exists()
    data = s2.get_details_for_id("ss", fl, False)
    assert fpath.exists()
    assert isinstance(data, dict)
    assert len(data) > 0
    assert "paperId" in data
    assert data["paperId"] in s2._rev_cache


def test_s2_cache_get_not_in_metadata_and_disk(s2):
    with open(os.path.join(s2._cache_dir, "metadata")) as f:
        metadata = f.read().split("\n")
    line = random.choice(metadata)
    metadata.remove(line)
    assert len(metadata) == 30
    with open(os.path.join(s2._cache_dir, "metadata"), "w") as f:
        f.write("\n".join(metadata))
    s2 = SemanticScholar(cache_dir="tests/cache_data/")
    key = line.split(",")[-1]
    # doesn't exist
    assert key not in s2._rev_cache
    fpath = s2._cache_dir.joinpath(key)
    if fpath.exists():
        os.remove(fpath)
    data = s2.get_details_for_id("ss", key, False)
    assert fpath.exists()
    assert isinstance(data, dict)
    assert len(data) > 0
    assert "paperId" in data
    # should exist now, called put
    assert data["paperId"] in s2._rev_cache


def test_s2_cache_get_not_ssid_not_in_metadata_and_disk(s2):
    arxiv_id = "2010.06775"
    with open(os.path.join(s2._cache_dir, "metadata")) as f:
        metadata = f.read().split("\n")
    if arxiv_id in s2._cache[s2.id_types("arxiv")]:
        key = s2._cache[s2.id_types("arxiv")][arxiv_id]
        metadata.remove(key)
        assert len(metadata) == 30
        with open(os.path.join(s2._cache_dir, "metadata"), "w") as f:
            f.write("\n".join(metadata))
    else:
        key = None
    s2 = SemanticScholar(cache_dir="tests/cache_data/")
    assert arxiv_id not in s2._cache[s2.id_types("arxiv")]
    if key:
        fpath = s2._cache_dir.joinpath(key)
        if fpath.exists():
            os.remove(fpath)
    data = s2.get_details_for_id("arxiv", arxiv_id, False)
    assert "paperId" in data
    fpath = s2._cache_dir.joinpath(data["paperId"])
    assert fpath.exists()
    assert isinstance(data, dict)
    assert len(data) > 0
    assert data["paperId"] in s2._rev_cache


def test_s2_details_fetches_correct_format_both_on_and_not_on_disk(s2, cache_files):
    fl = random.choice(cache_files)
    details = s2.get_details_for_id("ss", fl, False)
    assert "paperId" in details
    assert "citations" in details
    assert "references" in details
    fl = "5d9e7dbf28382eb3d8e1bbd2cae6a1c8d223ce4a"
    if fl in cache_files:
        os.remove(f"tests/cache_data/{fl}")
        cache_files.remove(fl)
    details = s2.get_details_for_id("ss", fl, False)
    assert "paperId" in details
    assert "citations" in details
    assert "references" in details


def test_s2_graph_search(s2):
    result = json.loads(s2.search("breiman random forests"))
    assert isinstance(result, dict)
    assert result["data"][0]["paperId"] == "13d4c2f76a7c1a4d0a71204e1d5d263a3f5a7986"


def test_s2_update_citations(s2):
    key = "13d4c2f76a7c1a4d0a71204e1d5d263a3f5a7986"
    if s2._cache_dir.joinpath(key).exists():
        os.remove(f"tests/cache_data/{key}")
    result = s2.details(key)
    result = s2._check_cache(key)
    assert "next" in result["citations"]
    assert len(result["citations"]["data"]) == s2._config["citations"]["limit"]
    next_citations = s2.next_citations(key)
    result = s2._check_cache(key)
    assert len(result["citations"]["data"]) == 2 * s2._config["citations"]["limit"]
    next_citations = s2.next_citations(key, 100)
    result = s2._check_cache(key)
    assert len(result["citations"]["data"]) == (2 * s2._config["citations"]["limit"]) + 100
