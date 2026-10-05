"""GET a JSON endpoint, retrying on rate limits and server errors."""

import time

import requests

MAX_TRIES = 5
RETRY_STATUS = {429, 500, 502, 503, 504}
TIMEOUT_S = 60


def get_json(url: str, params: dict) -> dict:
    """Return the parsed body, backing off 2, 4, 8, ... seconds between tries."""
    problem = ""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout) as err:
            problem = type(err).__name__
        else:
            if resp.status_code == 200:
                return resp.json()
            # never print resp.url, it carries the api key
            if resp.status_code not in RETRY_STATUS:
                raise RuntimeError(f"GET {url} failed with HTTP {resp.status_code}")
            problem = f"HTTP {resp.status_code}"

        if attempt < MAX_TRIES:
            wait = 2 ** attempt
            print(f"{problem} from {url}, retry {attempt} in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"GET {url} failed after {MAX_TRIES} tries ({problem})")
