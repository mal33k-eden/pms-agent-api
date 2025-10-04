"""
 * Author: Emmanuel Kwami Tartey
 * Date: 03 Oct, 2025
 * Time: 10:54 PM
 * Project: pms-agent
 * gitHub: https://github.com/mal33k-eden
"""
import asyncpg
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', '')


class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)

    async def disconnect(self):
        await self.pool.close()

    async def execute_schema(self):
        """Create all tables"""
        schema_path = Path(__file__).parent / 'init-schema.sql'
        schema = schema_path.read_text()

        async with self.pool.acquire() as conn:
            await conn.execute(schema)


db = Database()
