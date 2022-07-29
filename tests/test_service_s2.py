import pytest
import os
import json
import random
import yaml

import requests
from ref_man_py.semantic_scholar import SemanticScholar


# TODO: Different types of ids fetching
def test_server_get_s2_paper_arxiv(server):
    url = f"http://{server.host}:{server.port}"
    response = requests.get(f"{url}/s2_paper?id=2010.06775&id_type=arxiv")
    assert response.status_code == 200
    data = json.loads(response.content)
    assert isinstance(data, dict)
    assert "paperId" in data
    assert "externalIds" in data
    assert data["externalIds"]["ArXiv"] == "2010.06775"


def test_server_get_s2_paper(server):
    url = f"http://{server.host}:{server.port}"
    response = requests.get(f"{url}/s2_paper?id=96382611e8a8139df8771ea5c6b25d553cf3e9a5&id_type=ss")
    assert response.status_code == 200
    data = json.loads(response.content)
    assert isinstance(data, dict)
    assert "paperId" in data
    assert data["paperId"] == "96382611e8a8139df8771ea5c6b25d553cf3e9a5"


def test_server_get_s2_details(server):
    url = f"http://{server.host}:{server.port}"
    response = requests.get(f"{url}/s2_details/96382611e8a8139df8771ea5c6b25d553cf3e9a5")
    assert response.status_code == 200
    data = json.loads(response.content)
    assert isinstance(data, dict)
    assert "paperId" in data
    assert data["paperId"] == "96382611e8a8139df8771ea5c6b25d553cf3e9a5"


# def test_server_get_s2_all(server):
#     url = f"http://{server.host}:{server.port}"
#     response = requests.get(f"{url}/s2_all_details/96382611e8a8139df8771ea5c6b25d553cf3e9a5")
#     assert response.status_code == 200
#     data = json.loads(response.content)
#     assert isinstance(data, dict)
#     assert "details" in data and "citations" in data and "references" in data
#     assert "paperId" in data["details"]
#     assert data["details"]["paperId"] == "96382611e8a8139df8771ea5c6b25d553cf3e9a5"


def test_server_get_s2_citations_params(server):
    url = f"http://{server.host}:{server.port}"
    response = requests.get(f"{url}/s2_citations_params")
    assert response.status_code == 200
    data = json.loads(response.content)
    assert isinstance(data, dict)


def test_server_get_s2_citations_without_filters(server):
    url = f"http://{server.host}:{server.port}"
    key = "13d4c2f76a7c1a4d0a71204e1d5d263a3f5a7986"
    response = requests.post(f"{url}/s2_citations/{key}", json={})
    assert response.status_code == 200
    assert "method not implemented" in json.loads(response.content).lower()
    response = requests.get(f"{url}/s2_citations/{key}")
    assert response.status_code == 200
    data = json.loads(response.content)
    assert isinstance(data, list)
    assert all([isinstance(x, dict) and "paperId" in x for x in data])


def test_server_get_s2_citations_with_filters(server):
    url = f"http://{server.host}:{server.port}"
    key = "13d4c2f76a7c1a4d0a71204e1d5d263a3f5a7986"
    response = requests.post(f"{url}/s2_citations/{key}?count=100",
                             json={"filters": {"year": {"min_y": 2014, "max_y": 0},
                                               "num_citing": {"num": 10},
                                               "title": {"title_re": ".*covid.*", "invert": True}}})
    assert response.status_code == 200
    data = json.loads(response.content)
    assert isinstance(data, list)
    assert all([isinstance(x, dict) and "paperId" in x for x in data])
    assert all(["covid" not in x["title"].lower() for x in data])
    assert all([x["year"] >= 2014 for x in data])
    assert all([x["citationCount"] >= 10 for x in data])
    response = requests.post(f"{url}/s2_citations/{key}?count=5",
                             json={"filters": {"year": {"min_y": 2014, "max_y": 0},
                                               "num_citing": {"num": 10},
                                               "title": {"title_re": ".*covid.*", "invert": True}}})
    assert response.status_code == 200
    data = json.loads(response.content)
    assert isinstance(data, list)
    assert all([isinstance(x, dict) and "paperId" in x for x in data])
    assert all(["covid" not in x["title"].lower() for x in data])
    assert all([x["year"] >= 2014 for x in data])
    assert all([x["citationCount"] >= 10 for x in data])
    assert len(data) == 5


def test_server_get_s2_next_citations_when_have_next_citations(server):
    url = f"http://{server.host}:{server.port}"
    key = "13d4c2f76a7c1a4d0a71204e1d5d263a3f5a7986"
    response = requests.get(f"{url}/s2_citations/{key}")
    assert response.status_code == 200
    data_0 = json.loads(response.content)
    assert isinstance(data_0, list)
    assert all([isinstance(x, dict) and "paperId" in x for x in data_0])
    assert len(data_0) == 1000
    response = requests.get(f"{url}/s2_next_citations/{key}")
    assert response.status_code == 200
    data_1 = json.loads(response.content)
    assert len(data_1["data"]) == 1000
    response = requests.get(f"{url}/s2_citations/{key}?count=2000")
    data_2 = json.loads(response.content)
    assert len(data_2) == 2000
    response = requests.get(f"{url}/s2_next_citations/{key}?count=100")
    assert response.status_code == 200
    data_3 = json.loads(response.content)
    assert len(data_3["data"]) == 100
    response = requests.get(f"{url}/s2_citations/{key}?count=10000")
    data_4 = json.loads(response.content)
    assert len(data_4) == 2100


def test_server_get_s2_next_citations_when_dont_have_next_citations(server):
    url = f"http://{server.host}:{server.port}"
    key = "96382611e8a8139df8771ea5c6b25d553cf3e9a5"
    response = requests.get(f"{url}/s2_citations/{key}")
    assert response.status_code == 200
    data = json.loads(response.content)
    assert isinstance(data, list)
    response = requests.get(f"{url}/s2_next_citations/{key}")
    assert response.status_code == 200
    data = json.loads(response.content)
    assert data is None


# NOTE: This paper's ID has changed so have added both
def test_server_get_s2_search(server):
    query = "On the Generalization Mystery in Deep Learning"
    url = f"http://{server.host}:{server.port}"
    response = requests.get(f"{url}/s2_search?q={query}")
    result = json.loads(response.content)
    assert "error" not in result
    assert result["data"][0]["paperId"] in {"85a13ef1ca5a708a0860fdfb35361a55ff3b3d85",
                                            "1dcaaff675a61e39bf90ebb866bdb6d47161bcc5"}
