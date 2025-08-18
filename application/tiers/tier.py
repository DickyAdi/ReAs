from typing import Optional

from domain.entities.tiers.interfaces import TiersInterface
from domain.enums.tiers import Tier


class TierApplication:
    """Tier application for tier services."""

    def __init__(self, service: TiersInterface):
        self.service = service

    async def get_tier(
        self, db, id: Optional[int] = None, tier_name: Optional[Tier] = None
    ):
        """Get tier instances based on one of the given identifiers. Must only provide 1 identifier!

        Args:
            db (Any): Async session of the DB
            id (Optional[int], optional): Id of the tier. Defaults to None.
            tier_name (Optional[Tier], optional): Tier enum object. Defaults to None.

        Raises:
            ValueError: If both `id` and `tier_name` is provided. Choose only one.

        Returns:
            Tier: Infrastructure layer Tier ORM instances.
        """
        if id and tier_name:
            raise ValueError(
                "Only 1 identifiers must be provided, whether `id` or `tier_name`."
            )
        if id:
            tier_obj = await self.service.get_tier_by_id(db=db, id=id)
        else:
            tier_obj = await self.service.get_tier_by_name(db=db, tier_name=tier_name)
        return tier_obj
