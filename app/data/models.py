"""
 * Author: Emmanuel Kwami Tartey
 * Date: 03 Oct, 2025
 * Time: 11:09 PM
 * Project: pms-agent
 * gitHub: https://github.com/mal33k-eden
"""
from pydantic import BaseModel
from typing import List, Optional


class DrugSafetyResponse(BaseModel):
    drug_name: str
    pregnancy_category: Optional[str]
    pregnancy_safety: str
    breastfeeding_safety: str
    recommendations: str
    confidence: str
    warnings: Optional[List[str]] = []
