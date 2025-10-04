"""
 * Author: Emmanuel Kwami Tartey
 * Date: 03 Oct, 2025
 * Time: 10:55 PM
 * Project: pms-agent
 * gitHub: https://github.com/mal33k-eden
"""
import logging
import asyncio
from setup.db.config import db

log = logging.getLogger(__name__)


async def init():
    await db.connect()
    await db.execute_schema()
    log.info("Database initialized!")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(init())
