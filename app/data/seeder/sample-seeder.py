import logging
import asyncio
from setup.db.config import db

log = logging.getLogger(__name__)

COMMON_DRUGS = [
    {"name": "Tylenol", "generic": "acetaminophen", "category": "B", "bf_safety": "safe"},
    {"name": "Advil", "generic": "ibuprofen", "category": "C", "bf_safety": "moderate"},
    {"name": "Zoloft", "generic": "sertraline", "category": "C", "bf_safety": "moderate"},
    {"name": "Amoxicillin", "generic": "amoxicillin", "category": "B", "bf_safety": "safe"},
    {"name": "Benadryl", "generic": "diphenhydramine", "category": "B", "bf_safety": "moderate"},
]


async def seed():
    await db.connect()

    async with db.pool.acquire() as conn:
        for drug in COMMON_DRUGS:
            # Insert drug
            drug_id = await conn.fetchval("""
                                          INSERT INTO drugs (name, generic_name)
                                          VALUES ($1, $2) ON CONFLICT (name) DO
                                          UPDATE SET generic_name = $2
                                              RETURNING id
                                          """, drug["name"], drug["generic"])

            # Insert safety data
            await conn.execute("""
                               INSERT INTO drug_safety_data
                               (drug_id, pregnancy_category, pregnancy_safety,
                                breastfeeding_safety, ai_summary, data_source,
                                confidence_score)
                               VALUES ($1, $2, $3, $4, $5, $6, $7)
                               """,
                               drug_id,
                               drug["category"],
                               "safe" if drug["category"] in ["A", "B"] else "caution",
                               drug["bf_safety"],
                               f"{drug['name']} is generally considered {drug['bf_safety']} during breastfeeding.",
                               "manual",
                               0.9
                               )

    log.info("Seed data inserted!")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(seed())
