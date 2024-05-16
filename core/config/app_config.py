import importlib
import os

from app import app

for filename in os.listdir("api/routers"):
    if "_router.py" not in filename:
        continue

    module = importlib.import_module("api.routers." + filename.split(".")[0])
    if hasattr(module, "router"):
        app.include_router(module.router)