import requests
import re
from urllib.parse import urlparse, urlencode
from typing import Optional

from domain.externals.scraping import ScrapingInterface
from config.settings import settings


class ScrapingService(ScrapingInterface):
    def build(
        self,
        query_params: dict,
        resource_url: str,
        base_url: Optional[str] = None,
    ):
        base_query_params = {"ignoreEmpty": True, "reviewsLimit": 0, "async": True}
        base_url = (
            "https://api.outscraper.cloud/maps/reviews-v3" if not base_url else base_url
        )
        if "query" not in list(query_params.keys()):
            cid = self.get_cid(gmaps_url=resource_url)
            query_params["query"] = cid
            query_params = {**base_query_params, **query_params}
        self.resource = f"{base_url}?{urlencode(query_params)}"

    def hit(self, headers: Optional[dict] = None):
        if not self.resource:
            raise ValueError(f"Resource must not be None. Got {type(self.resource)}")
        headers = {
            "X-API-KEY": f"{settings.scraping_auth_key}" if not headers else headers
        }
        return requests.get(self.resource, stream=True, headers=headers)

    def get_cid(self, gmaps_url: str) -> str:
        parsed_url = urlparse(gmaps_url)
        if parsed_url.netloc != "www.google.com":
            tobe_fix = requests.head(gmaps_url, allow_redirects=True)
            fix_url = tobe_fix.url
        else:
            fix_url = gmaps_url

        if patt := re.search(r"0x[0-9a-zA-Z]+:0x[0-9a-zA-Z]+", fix_url):
            cidHex = patt.group().split(":")[-1]
            return str(int(cidHex, 16))
        else:
            raise ValueError(
                f"Cannot or dont know to obtain google maps CID from the given url. Url is {gmaps_url}"
            )
