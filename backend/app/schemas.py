from pydantic import BaseModel, Field
from typing import Optional

   

class TopPrediction(BaseModel):
    category: str
    probability: float


class ScreenResponse(BaseModel):
    filename: str
    predicted_category: str
    confidence: float
    top_predictions: list[TopPrediction]
    match_score: Optional[float] = Field(
        default=None,
        description="Similarity %  against the job description, if provided",
    )
    matched_keywords: Optional[list[str]] = None
    
