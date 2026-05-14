from typing import Optional
from pydantic import BaseModel, Field


# ── 기존 스키마 유지 ───────────────────────────────────────────────────────────

class MotionResult(BaseModel):
    is_correct: bool
    score: int = Field(ge=0, le=100)
    feedback: str
    exercise_type: str


class SessionSummary(BaseModel):
    total_reps: int
    avg_score: float
    duration_seconds: int


# ── 산책 AI 대화 ───────────────────────────────────────────────────────────────

class WalkStartRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    session_id: Optional[str] = None   # 신규: 세션 연속성


class WalkStartResponse(BaseModel):
    message: str
    session_id: str                    # 신규: 클라이언트가 세션 ID 저장
    audio_base64: Optional[str] = None # 신규: TTS 오디오 (base64 MP3)


class WalkChatResponse(BaseModel):
    user_text: Optional[str] = None    # STT 정규화 결과 또는 텍스트 입력
    raw_stt: Optional[str] = None      # STT 원본 (발음 보정 전, 디버그용)
    ai_text: str                       # LLM 응답 텍스트
    audio_base64: Optional[str] = None # TTS 오디오 (base64 MP3)
    corrections: list[dict] = Field(default_factory=list)  # 발음 보정 목록


# ── 스트리밍 ──────────────────────────────────────────────────────────────────

class WalkStreamRequest(BaseModel):
    session_id: str
    messages: list[dict] = Field(default_factory=list)
    text: str


# ── 세션 종료 ─────────────────────────────────────────────────────────────────

class WalkEndRequest(BaseModel):
    session_id: str


class WalkEndResponse(BaseModel):
    ok: bool
    message: str  # "요약 중" or "대화가 너무 짧아 요약을 건너뜁니다"


# ── 메모리 ────────────────────────────────────────────────────────────────────

class MemorySummaryResponse(BaseModel):
    long_term_summary: str
    top_topics: list[str]
    recent_emotions: list[str]
