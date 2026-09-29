import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()
from src.router.router import router

app = FastAPI(title="DAMO RADAR Abdominal Service")
app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8003))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)