import time
import logging
import requests
from .auth import load_private_key, sign_request

logger = logging.getLogger(__name__)

API_PREFIX = "/trade-api/v2"


class KalshiClient:
    def __init__(self, key_id: str, private_key_path: str, base_url: str):
        self.key_id = key_id
        self.base_url = base_url
        self.private_key = load_private_key(private_key_path)
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def _auth_headers(self, method: str, endpoint: str) -> dict:
        full_path = f"{API_PREFIX}{endpoint}"
        timestamp, signature = sign_request(self.private_key, method, full_path)
        return {
            'KALSHI-ACCESS-KEY': self.key_id,
            'KALSHI-ACCESS-TIMESTAMP': timestamp,
            'KALSHI-ACCESS-SIGNATURE': signature,
        }

    def get(self, endpoint: str, params: dict = None, auth: bool = True) -> dict:
        headers = self._auth_headers('GET', endpoint) if auth else {}
        for attempt in range(3):
            try:
                resp = self.session.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers=headers,
                    timeout=15
                )
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"GET {endpoint} attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(1)

    def post(self, endpoint: str, body: dict) -> dict:
        headers = self._auth_headers('POST', endpoint)
        resp = self.session.post(
            f"{self.base_url}{endpoint}",
            json=body,
            headers=headers,
            timeout=15
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self, endpoint: str) -> dict:
        headers = self._auth_headers('DELETE', endpoint)
        resp = self.session.delete(
            f"{self.base_url}{endpoint}",
            headers=headers,
            timeout=15
        )
        resp.raise_for_status()
        return resp.json()
