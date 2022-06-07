import pytest
import requests
import json
import yaml


def test_service_echo(server):
    url = f"http://{server.host}:{server.port}"
    response = requests.get(f"{url}/echo")
    assert response.status_code == 200
    assert response.content.decode("utf-8") == "echo"


def test_service_version(server):
    url = f"http://{server.host}:{server.port}"
    response = requests.get(f"{url}/version")
    assert response.status_code == 200


def test_service_get_yaml(server):
    url = f"http://{server.host}:{server.port}"
    response = requests.post(f"{url}/get_yaml", json={"key": "value"})
    assert response.status_code == 200
    assert yaml.load(response.content, Loader=yaml.FullLoader)


def test_service_get_arxiv(server):
    url = f"http://{server.host}:{server.port}"
    response = requests.get(f"{url}/arxiv?id=2010.06775")
    assert response.status_code == 200
    assert json.loads(response.content).startswith("@")


def test_service_dblp(server):
    pass


def test_service_get_cvf_url(server):
    pass


def test_service_get_update_cache(server):
    pass


def test_service_force_stop_update_cache(server):
    pass


def test_service_fetch_url_no_proxy(server):
    pass


def test_service_fetch_url_with_proxy(server):
    pass


def test_service_fetch_url_progress(server):
    pass


def test_service_url_info(server):
    pass
