"""SQLAlchemy database engine and session setup — works with SQLite and PostgreSQL."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config import settings


_IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")

_engine_kwargs = {}
if _IS_SQLITE:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=10)

engine = create_engine(settings.DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def init_db():
    """Create all tables. Import models first to register them on Base.
    Also adds any missing columns for existing tables (lightweight migration)."""
    from sqlalchemy import inspect, text
    import backend.models  # noqa: F401 — register ORM models
    Base.metadata.create_all(bind=engine)

    # Lightweight migration: add columns that were added after initial migration
    _MIGRATIONS = {
        "model_configs": [
            ("single_cls", "BOOLEAN DEFAULT FALSE"),
            ("cos_lr", "BOOLEAN DEFAULT FALSE"),
            ('"freeze"', "INTEGER"),
            ("close_mosaic", "INTEGER DEFAULT 10"),
        ],
    }
    insp = inspect(engine)
    with engine.connect() as conn:
        for table, cols in _MIGRATIONS.items():
            if table not in insp.get_table_names():
                continue
            existing_cols = {c["name"] for c in insp.get_columns(table)}
            for col_name, col_def in cols:
                # Strip quotes for the existence check
                check_name = col_name.strip('"')
                if check_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}"))
                    conn.commit()
