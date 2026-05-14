"""
산책 AI 대화 라우터 — 온디바이스 파이프라인

엔드포인트:
  POST /api/v1/walk/start        기존 유지 — 날씨 기반 인사말 (TTS 포함)
  POST /api/v1/walk/chat         기존 유지 — STT+LLM+TTS 풀 파이프라인
  GET  /api/v1/walk/stream       신규 — SSE 스트리밍 LLM 응답
  GET  /api/v1/walk/memory       신규 — 유저 메모리 요약 조회

기존 라우터 구조/rate limit/auth 완전 유지.
"""

import base64
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.middleware.auth import verify_firebase_token
from app.models.schemas import (
    MemorySummaryResponse,
    WalkChatResponse,
    WalkEndRequest,
    WalkEndResponse,
    WalkStartRequest,
    WalkStartResponse,
)
from app.services import memory_service
from app.services.walk_ai_service import (
    end_session,
    generate_walk_greeting,
    process_chat_full,
    stream_chat,
    transcribe_audio,
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)

MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_AUDIO = {
    "audio/m4a", "audio/mp4", "audio/mpeg", "audio/wav",
    "audio/webm", "audio/ogg", "video/mp4",
}
ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp"}


# ── /walk/start ───────────────────────────────────────────────────────────────

@router.post("/walk/start", response_model=WalkStartResponse)
@limiter.limit("10/minute")
async def walk_start(
    request: Request,
    body: WalkStartRequest,
    user=Depends(verify_firebase_token),
):
    """산책 시작 — 날씨 기반 인사말 + TTS 오디오 반환."""
    uid: str = user.get("uid", "anonymous")

    # 세션 생성 (클라이언트에서 session_id 미전달 시 서버에서 발급)
    session_id = body.session_id or await memory_service.create_session(uid)

    greeting = await generate_walk_greeting(body.lat, body.lon, uid=uid)

    # TTS 합성
    audio_b64: Optional[str] = None
    try:
        from app.services.tts_service import synthesize_with_timeout
        audio_bytes = await synthesize_with_timeout(greeting)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.warning("TTS failed for walk/start: %s", e)

    # 인사말을 assistant 메시지로 저장
    await memory_service.save_message(session_id, uid, "assistant", greeting)

    return WalkStartResponse(
        message=greeting,
        session_id=session_id,
        audio_base64=audio_b64,
    )


# ── /walk/chat ────────────────────────────────────────────────────────────────

@router.post("/walk/chat", response_model=WalkChatResponse)
@limiter.limit("30/minute")
async def walk_chat(
    request: Request,
    messages: str = Form(..., description="대화 기록 JSON (List[{role, content}])"),
    session_id: str = Form(..., description="세션 ID (/walk/start 에서 발급된 값)"),
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    include_tts: bool = Form(True, description="TTS 오디오 포함 여부"),
    user=Depends(verify_firebase_token),
):
    """
    텍스트/음성 → STT → 발음정규화 → LLM → TTS 풀 파이프라인.

    - text: 직접 텍스트 입력 (audio보다 우선)
    - audio: 음성 파일 (faster-whisper STT)
    - image: 사진 (현재 텍스트 처리로 대체)
    - include_tts: False면 응답 속도 빠름 (오디오 미포함)
    """
    uid: str = user.get("uid", "anonymous")

    # ── messages 파싱 ─────────────────────────────────────────────────────────
    try:
        message_list: list[dict] = json.loads(messages)
        if not isinstance(message_list, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="messages가 유효한 JSON 배열이 아닙니다.")

    # ── 입력 텍스트 ───────────────────────────────────────────────────────────
    user_text: Optional[str] = text.strip() if text and text.strip() else None

    # ── 오디오 처리 ───────────────────────────────────────────────────────────
    audio_bytes: Optional[bytes] = None
    audio_filename = "audio.m4a"
    if not user_text and audio:
        if audio.content_type not in ALLOWED_AUDIO:
            raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식입니다.")
        audio_bytes = await audio.read()
        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=400, detail="오디오 파일 크기는 10MB 이하여야 합니다.")
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="오디오 파일이 비어 있습니다.")
        audio_filename = audio.filename or "audio.m4a"

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

    if not user_text and not audio_bytes and not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="text, audio, image 중 하나 이상을 입력해야 합니다.",
        )

    # ── 풀 파이프라인 실행 ────────────────────────────────────────────────────
    try:
        result = await process_chat_full(
            session_id=session_id,
            uid=uid,
            messages=message_list,
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
            text_input=user_text,
            image_bytes=image_bytes,
            image_media_type=image_media_type,
            include_tts=include_tts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── TTS 오디오 base64 인코딩 ──────────────────────────────────────────────
    audio_b64: Optional[str] = None
    if result["audio_bytes"]:
        audio_b64 = base64.b64encode(result["audio_bytes"]).decode("utf-8")

    return WalkChatResponse(
        user_text=result["user_text"],
        raw_stt=result["raw_stt"],
        ai_text=result["ai_text"],
        audio_base64=audio_b64,
        corrections=result["corrections"],
    )


# ── /walk/stream (신규 SSE) ────────────────────────────────────────────────────

@router.get("/walk/stream")
@limiter.limit("20/minute")
async def walk_stream(
    request: Request,
    session_id: str = Query(...),
    text: str = Query(...),
    messages: str = Query(default="[]"),
    user=Depends(verify_firebase_token),
):
    """
    Server-Sent Events (SSE) 스트리밍 LLM 응답.
    토큰 단위 실시간 응답 — TTS 미포함.

    클라이언트 사용법 (React Native EventSource 또는 fetch stream):
      GET /api/v1/walk/stream?session_id=...&text=안녕하세요&messages=[]
      Authorization: Bearer <firebase_token>
    """
    uid: str = user.get("uid", "anonymous")

    try:
        message_list: list[dict] = json.loads(messages)
    except (json.JSONDecodeError, ValueError):
        message_list = []

    async def event_generator():
        try:
            async for token in stream_chat(
                session_id=session_id,
                uid=uid,
                messages=message_list,
                user_text=text,
            ):
                # SSE 형식: "data: <token>\n\n"
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            # 스트림 종료 신호
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("SSE stream error for uid=%s: %s", uid, e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        },
    )


# ── /walk/end (신규) ──────────────────────────────────────────────────────────

@router.post("/walk/end", response_model=WalkEndResponse)
@limiter.limit("30/minute")
async def walk_end(
    request: Request,
    body: WalkEndRequest,
    user=Depends(verify_firebase_token),
):
    """
    산책 세션 종료.
    - DB에 종료 시간 기록
    - 대화 내용을 백그라운드에서 LLM으로 요약 → long_term_summary 누적
    - 응답은 즉시 반환 (요약은 백그라운드 처리)
    """
    uid: str = user.get("uid", "anonymous")

    # 세션에 속한 유저 메시지 수 확인 (너무 짧으면 건너뜀)
    messages = await memory_service.get_session_messages(body.session_id)
    user_turns = [m for m in messages if m["role"] == "user"]

    await end_session(body.session_id, uid)

    if len(user_turns) < memory_service._MIN_MESSAGES_TO_SUMMARIZE:
        return WalkEndResponse(
            ok=True,
            message="대화가 너무 짧아 요약을 건너뜁니다.",
        )

    return WalkEndResponse(
        ok=True,
        message=f"오늘 대화 {len(user_turns)}턴을 기억하고 있을게요. 다음 산책 때 반영됩니다.",
    )


# ── /walk/memory (신규) ───────────────────────────────────────────────────────

@router.get("/walk/memory", response_model=MemorySummaryResponse)
@limiter.limit("20/minute")
async def get_memory(
    request: Request,
    user=Depends(verify_firebase_token),
):
    """유저의 메모리 요약 조회 (앱 UI 표시용)."""
    uid: str = user.get("uid", "anonymous")

    summary = await memory_service.get_long_term_summary(uid)
    top_topics = await memory_service.get_top_topics(uid)
    recent_emotions = await memory_service.get_recent_emotions(uid)

    return MemorySummaryResponse(
        long_term_summary=summary,
        top_topics=top_topics,
        recent_emotions=recent_emotions,
    )
