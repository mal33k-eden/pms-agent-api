"""
 * Author: Emmanuel Kwami Tartey
 * Date: 03 Oct, 2025
 * Time: 11:51 PM
 * Project: pms-agent
 * gitHub: https://github.com/mal33k-eden
"""
import pytest
import httpx
import asyncio


@pytest.mark.asyncio
async def test_known_drug():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/api/drug/tylenol")
        assert response.status_code == 200
        data = response.json()
        assert data["drug_name"] == "Tylenol"
        assert data["pregnancy_category"] == "B"


if __name__ == "__main__":
    asyncio.run(test_known_drug())
