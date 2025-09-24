# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from decimal import Decimal
from uuid import uuid4

from config.settings import settings

from domain.enums.tiers import Tier
from domain.enums.tiers_pricing import Currency
from domain.enums.subscriptions import SubscriptionPlan
from domain.enums.users import Role, AuthProvider

from application.use_cases.subscription import UserRegistrationFlow

from domain.entities.subscriptions import SubscriptionPlansEntity
from domain.entities.tiers import TierEntity

from infrastructure.db.services import (
    TierService,
    SubscriptionPlanService,
    UserService,
    SubscriptionService,
    TransactionService,
    PaymentService,
)
from infrastructure.db import Base  # noqa


async def get_seeder_session(connection) -> AsyncSession:
    SessionLocal = async_sessionmaker(
        bind=connection, autoflush=False, expire_on_commit=False
    )
    db = SessionLocal()
    return db


async def seed_subscription_plan_n_tier(db):
    list_of_tier = [Tier.Base, Tier.Mid, Tier.Pro]
    list_of_plans = [
        {
            "tier": Tier.Base,
            "cycle": SubscriptionPlan.free,
            "currency": Currency.idr,
            "price": Decimal(0),
            "duration_days": None,
            "is_active": True,
        },
        {
            "tier": Tier.Mid,
            "cycle": SubscriptionPlan.monthly,
            "currency": Currency.idr,
            "price": Decimal(40000),
            "duration_days": 30,
            "is_active": True,
        },
        {
            "tier": Tier.Mid,
            "cycle": SubscriptionPlan.annually,
            "currency": Currency.idr,
            "price": Decimal(400000),
            "duration_days": 365,
            "is_active": True,
        },
        {
            "tier": Tier.Pro,
            "cycle": SubscriptionPlan.monthly,
            "currency": Currency.idr,
            "price": Decimal(80000),
            "duration_days": 30,
            "is_active": True,
        },
        {
            "tier": Tier.Pro,
            "cycle": SubscriptionPlan.annually,
            "currency": Currency.idr,
            "price": Decimal(800000),
            "duration_days": 365,
            "is_active": True,
        },
    ]
    tiers = []
    for tier in list_of_tier:
        created = TierEntity.create(name=tier)
        tiers.append(created)

    tier_service = TierService()
    plan_service = SubscriptionPlanService()
    for tier in tiers:
        created_tier = await tier_service.create_tier(db=db, tier=tier, commit=False)
        for plan in list_of_plans:
            if plan.get("tier") == tier.name:
                plan_obj = SubscriptionPlansEntity.create(
                    **plan, tier_id=created_tier.id
                )
                created_plan = await plan_service.create_plans(
                    db=db, plan=plan_obj, commit=False
                )
        db.add_all([created_tier, created_plan])
        print("[SEED] Seed subscription finished.")


async def seed_user_admin_superadmin(db: AsyncSession):
    flow = UserRegistrationFlow(
        user_service=UserService(),
        subscription_service=SubscriptionService(),
        payment_service=PaymentService(),
        plan_service=SubscriptionPlanService(),
        transaction_service=TransactionService(),
    )
    admin_creds = {
        "name": "Admin1",
        "email": settings.admin_email,
        "password": settings.admin_password,
        "is_validated": True,
        "role": Role.admin,
        "provider": AuthProvider.local,
    }
    superadmin_creds = {
        "name": "Superadmin1",
        "email": settings.superadmin_email,
        "password": settings.superadmin_password,
        "is_validated": True,
        "role": Role.superadmin,
        "provider": AuthProvider.local,
    }
    user_creds = {
        "name": "User1",
        "email": "user@example.com",
        "password": "mypassword123",
        "is_validated": True,
        "role": Role.user,
        "provider": AuthProvider.local,
    }
    create_superadmin = await flow(  # noqa
        db=db,
        name=superadmin_creds["name"],
        email=superadmin_creds["email"],
        password=superadmin_creds["password"],
        provider=superadmin_creds["provider"],
        idempotency_key=uuid4(),
        is_validated=superadmin_creds["is_validated"],
        role=superadmin_creds["role"],
        commit=False,
    )
    create_admin = await flow(  # noqa
        db=db,
        name=admin_creds["name"],
        email=admin_creds["email"],
        password=admin_creds["password"],
        provider=admin_creds["provider"],
        idempotency_key=uuid4(),
        is_validated=admin_creds["is_validated"],
        role=admin_creds["role"],
        commit=False,
    )
    _create_user = await flow(
        db=db,
        name=user_creds["name"],
        email=user_creds["email"],
        password=user_creds["password"],
        provider=user_creds["provider"],
        idempotency_key=uuid4(),
        is_validated=user_creds["is_validated"],
        role=user_creds["role"],
        commit=False,
    )
    print("[SEED] Seed admin & superadmin finished.")


# ? Unused function, commented out in case needed in the future. Current approach is dropping all alembic created table schemas.
# async def migrate_refresh(connection):
#     db = get_seeder_session(connection=connection)
#     table_names = list(Base.metadata.tables.keys())
#     if not table_names:
#         print("[SEED] Detected no tables. Breaking refresh.")
#         return
#     tables = ', '.join(f'"{name}"' for name in table_names)
#     stmnt = text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE;")
#     await db.execute(statement=stmnt)
#     await db.commit()
#     print('[SEED] Refresh finished.')


async def run_seeder(db: AsyncSession):
    async with db.begin():
        await seed_subscription_plan_n_tier(db=db)
        await seed_user_admin_superadmin(db=db)
    print("[SEED] All seeding finished.")


async def run(connection):
    db = await get_seeder_session(connection)
    try:
        await run_seeder(db=db)
    finally:
        await db.close()
