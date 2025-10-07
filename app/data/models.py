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
