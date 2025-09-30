from typing import Optional


from domain.externals.scraping import ScrapingInterface


class ScrapingApplication:
    def __init__(self, service: ScrapingInterface):
        self.service = service

    def hit(self, headers: Optional[dict] = None):
        data = self.service.hit(headers=headers)
        return data

    def build_hit(
        self,
        resource_url: str,
        base_url: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        data = self.service.build_hit(
            base_url=base_url,
            resource_url=resource_url,
            headers=headers,
        )
        return data
