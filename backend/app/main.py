from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import resume

app = FastAPI(
    title="Resume Screening System",
    description="Upload a resume, get an automatic job-category prediction "
                "and (optionally) a match score against a job description.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume.router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Resume Screening API is running"}
