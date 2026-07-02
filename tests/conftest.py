"""Test-time environment defaults.

The app's pydantic ``Settings`` requires a handful of env vars that are normally
supplied by a sourced ``.env`` on the CPU box. Set harmless local defaults here
(conftest is imported before any test module) so the suite runs without infra.
Real env (if exported) still wins — we only ``setdefault``.
"""
import os

_DEFAULTS = {
    "MONGODB_URL": "mongodb://localhost:27017",
    "MONGODB_DB_NAME": "World_builder_test",
    "MONGO_DB": "World_builder_test",
    "CELERY_BROKER_URL": "redis://localhost:6379/0",
    "CELERY_RESULT_BACKEND": "redis://localhost:6379/0",
    "AWS_S3_BUCKET": "test-bucket",
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)
