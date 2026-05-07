from typing import Optional
from pydantic import BaseModel, Field


class MotionResult(BaseModel):
    is_correct: bool
    score: int = Field(ge=0, le=100)
    feedback: str
    exercise_type: str


class SessionSummary(BaseModel):
    total_reps: int
    avg_score: float
    duration_seconds: int


# ─── 산책 AI 대화 ─────────────────────────────────────────────────────────────

class WalkStartRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None


class WalkStartResponse(BaseModel):
    message: str


class WalkChatResponse(BaseModel):
    user_text: Optional[str] = None   # STT 결과 (음성 첨부 시)
    ai_text: str                       # LLM 응답 (TTS로 재생)

