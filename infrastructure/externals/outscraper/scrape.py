import requests
import re
from urllib.parse import urlparse, urlencode
from typing import Optional

from domain.externals.scraping import ScrapingInterface
from config.settings import settings


class ScrapingService(ScrapingInterface):
    def build(
        self,
        resource_url: str,
        base_url: str = "https://api.outscraper.cloud/maps/reviews-v3",
    ):
        query_params = {
            "query": self.get_cid(resource_url),
            "ignoreEmpty": True,
            "reviewsLimit": 0,
            "async": True,
            "language": "id",
        }
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
