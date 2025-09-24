from infrastructure.db import SessionLocal, UnitOfWork


class SessionFactory:
    @staticmethod
    def create() -> UnitOfWork:
        sess = SessionLocal()
        return UnitOfWork(db=sess)
