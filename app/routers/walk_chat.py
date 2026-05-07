"""
산책 AI 대화 라우터
- POST /api/v1/walk/start  : 산책 시작 → 날씨 기반 인사말 생성
- POST /api/v1/walk/chat   : 텍스트/STT(음성) → LLM(대화) → 텍스트 응답
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.middleware.auth import verify_firebase_token
from app.models.schemas import WalkChatResponse, WalkStartRequest, WalkStartResponse
from app.services.walk_ai_service import (
    generate_walk_greeting,
    process_chat,
    transcribe_audio,
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_SIZE = 5 * 1024 * 1024   # 5 MB
ALLOWED_AUDIO = {
    "audio/m4a", "audio/mp4", "audio/mpeg", "audio/wav",
    "audio/webm", "audio/ogg", "video/mp4",
}
ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp"}


# ─── /walk/start ─────────────────────────────────────────────────────────────

@router.post("/walk/start", response_model=WalkStartResponse)
@limiter.limit("10/minute")
async def walk_start(
    request: Request,
    body: WalkStartRequest,
    user=Depends(verify_firebase_token),
):
    """실외 산책 시작 — 날씨 정보를 포함한 AI 인사말 반환"""
    greeting = await generate_walk_greeting(body.lat, body.lon)
    return WalkStartResponse(message=greeting)


# ─── /walk/chat ──────────────────────────────────────────────────────────────

@router.post("/walk/chat", response_model=WalkChatResponse)
@limiter.limit("30/minute")
async def walk_chat(
    request: Request,
    messages: str = Form(..., description="대화 기록 JSON (List[{role, content}])"),
    text: Optional[str] = Form(None, description="사용자 텍스트 입력 (STT 대체)"),
    audio: Optional[UploadFile] = File(None, description="음성 파일 (Whisper STT, API 지원금 후 활성화)"),
    image: Optional[UploadFile] = File(None, description="사진 파일 (멀티모달)"),
    user=Depends(verify_firebase_token),
):
    """
    텍스트(우선) 또는 음성(선택) + 사진(선택) → LLM 대화 → 텍스트 응답

    messages: JSON 배열 문자열 예시
      [{"role": "assistant", "content": "안녕하세요!"}, ...]
    """
    # ── 대화 기록 파싱 ────────────────────────────────────────────────────────
    try:
        message_list: list[dict] = json.loads(messages)
        if not isinstance(message_list, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="messages가 유효한 JSON 배열이 아닙니다.")

    # ── 텍스트 직접 입력 (현재 모드) ─────────────────────────────────────────
    user_text: Optional[str] = text.strip() if text and text.strip() else None

    # ── STT: 음성 → 텍스트 (OPENAI_API_KEY 설정 후 활성화) ───────────────────
    if not user_text and audio:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=503, detail="음성 입력은 준비 중입니다. 텍스트로 입력해 주세요.")
        if audio.content_type not in ALLOWED_AUDIO:
            raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식입니다.")
        audio_bytes = await audio.read()
        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=400, detail="오디오 파일 크기는 10MB 이하여야 합니다.")
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="오디오 파일이 비어 있습니다.")
        user_text = await transcribe_audio(audio_bytes, audio.filename or "audio.m4a")

    # ── 이미지 처리 ───────────────────────────────────────────────────────────
    image_bytes: Optional[bytes] = None
    image_media_type: Optional[str] = None
    if image:
        if image.content_type not in ALLOWED_IMAGE:
            raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다.")
        image_bytes = await image.read()
        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="이미지 파일 크기는 5MB 이하여야 합니다.")
        image_media_type = image.content_type

    # 텍스트도 이미지도 없는 경우
    if not user_text and not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="text 또는 image 중 하나 이상을 입력해야 합니다.",
        )

    # ── LLM 대화 생성 ─────────────────────────────────────────────────────────
    ai_text = await process_chat(
        messages=message_list,
        user_text=user_text,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
    )

    return WalkChatResponse(user_text=user_text, ai_text=ai_text)
