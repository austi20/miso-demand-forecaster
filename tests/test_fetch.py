"""Tests for the retrying JSON GET."""

import pytest
import requests

from src import fetch


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self.body = body

    def json(self):
        return self.body


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)


def fake_get_from(responses, calls):
    def fake_get(url, params, timeout):
        calls.append(params)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    return fake_get


def test_returns_body_on_success(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch.requests, "get", fake_get_from([FakeResponse(200, {"ok": 1})], calls))
    assert fetch.get_json("https://example.com", {}) == {"ok": 1}
    assert len(calls) == 1


def test_retries_rate_limit_and_connection_errors(monkeypatch):
    calls = []
    responses = [FakeResponse(429), requests.ConnectionError("down"), FakeResponse(200, {"ok": 1})]
    monkeypatch.setattr(fetch.requests, "get", fake_get_from(responses, calls))
    assert fetch.get_json("https://example.com", {}) == {"ok": 1}
    assert len(calls) == 3


def test_client_error_fails_fast_without_leaking_key(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch.requests, "get", fake_get_from([FakeResponse(403)], calls))
    with pytest.raises(RuntimeError) as err:
        fetch.get_json("https://example.com", {"api_key": "SECRET123"})
    assert len(calls) == 1
    assert "403" in str(err.value)
    assert "SECRET123" not in str(err.value)


def test_gives_up_after_max_tries(monkeypatch):
    calls = []
    responses = [FakeResponse(503) for _ in range(fetch.MAX_TRIES)]
    monkeypatch.setattr(fetch.requests, "get", fake_get_from(responses, calls))
    with pytest.raises(RuntimeError):
        fetch.get_json("https://example.com", {})
    assert len(calls) == fetch.MAX_TRIES
