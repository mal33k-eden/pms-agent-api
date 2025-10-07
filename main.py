from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.routes.drug import router as drug_router
from setup.db.config import db
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.connect()
    await db.execute_schema()
    yield
    # Shutdown
    await db.disconnect()


app = FastAPI(title="Drug Safety API", lifespan=lifespan)
app.include_router(drug_router)


@app.get("/")
async def root():
    return {"message": "PMS AGENT API", "version": "1.0"}
