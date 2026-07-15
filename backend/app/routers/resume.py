from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from app.services.parser import extract_text, UnsupportedFileType
from app.services.classifier import predict_category
from app.services.matcher import match_score
from app.schemas import ScreenResponse

router = APIRouter(prefix="/api", tags=["resume"])

MAX_FILE_SIZE_MB = 5

@router.post("/screen", response_model=ScreenResponse)
async def screen_resume( file: UploadFile = File(..., description="Resume file (.pdf or .docx)"),
    job_description: Optional[str] = Form(None, description="Optional job description text to compute a match score against" )):

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    try:
        resume_text = extract_text(file.filename, file_bytes)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not resume_text or not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from this file. It may be a scanned/image-based PDF.",
        )

    classification = predict_category(resume_text)

    result = {
        "filename": file.filename,
        "predicted_category": classification["predicted_category"],
        "confidence": classification["confidence"],
        "top_predictions": classification["top_predictions"],
        "match_score": None,
        "matched_keywords": None
    }

    if job_description and job_description.strip():
        match = match_score(resume_text, job_description)
        result["match_score"] = match["match_score"]
        result["matched_keywords"] = match["matched_keywords"]

    return result
