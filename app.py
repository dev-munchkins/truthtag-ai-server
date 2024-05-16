from fastapi import FastAPI
from core.config.app_config import *

app = FastAPI(
    title="TruthTag AI API",
    version="0.0.1",
    license_info={"name": "MIT License", "identifier": "MIT"}
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app="app:app",
        port=8000,
        reload=True,
    )
