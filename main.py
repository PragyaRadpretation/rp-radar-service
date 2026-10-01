import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()
from src.router.router import router
from utils.utils import warmup_radar_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load and warm up the model before receiving requests
    print("[*] Starting service and warming up DAMO RADAR model...")
    warmup_radar_model()
    yield
    # Shutdown logic (if any cleanup is needed when stopping)
    print("[*] Shutting down DAMO RADAR Abdominal Service...")


app = FastAPI(
    title="DAMO RADAR Abdominal Service",
    lifespan=lifespan
)

app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7017))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)