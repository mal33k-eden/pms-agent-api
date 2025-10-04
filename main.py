from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from setup.db.config import db
from app.services.fda_client import FDAClient
from app.data.models import DrugSafetyResponse
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
fda_client = FDAClient()


@app.get("/")
async def root():
    return {"message": "Drug Safety API", "version": "1.0"}


@app.get("/api/drug/{drug_name}", response_model=DrugSafetyResponse)
async def get_drug_safety(drug_name: str):
    # Log search
    # await log_search(drug_name)

    # Check database first
    drug_data = await get_from_database(drug_name)
    if drug_data:
        return drug_data

    # Not in DB - continue to fetch from FDA flow


async def log_search(search_term: str):
    async with db.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO searches (search_term, found) VALUES ($1, $2)",
            search_term,
            False
        )


async def get_from_database(drug_name: str):
    async with db.pool.acquire() as conn:
        result = await conn.fetchrow("""
                                     SELECT d.*, ds.*
                                     FROM drugs d
                                              JOIN drug_safety_data ds ON d.id = ds.drug_id
                                     WHERE LOWER(d.name) = LOWER($1)
                                        OR LOWER(d.generic_name) = LOWER($1)
                                         AND ds.expires_at > NOW()
                                         LIMIT 1
                                     """, drug_name)

        if result:
            return DrugSafetyResponse(
                drug_name=result['name'],
                pregnancy_category=result['pregnancy_category'],
                pregnancy_safety=result['pregnancy_safety'],
                breastfeeding_safety=result['breastfeeding_safety'],
                recommendations=result['ai_summary'],
                confidence="high" if result['confidence_score'] > 0.7 else "moderate"
            )

    return None
