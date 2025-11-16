from pydantic import BaseModel
from typing import List, Optional


class DrugSafetyResponse(BaseModel):
    drug_name: str
    pregnancy_safety: str
    breastfeeding_safety: str
    recommendations: str
    confidence: str
    warnings: Optional[List[str]] = [],
    study_count: Optional[int] = 0,
    data_source: Optional[str] = 'fda'
    analysis_type: Optional[str] = 'basic'
