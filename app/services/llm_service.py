"""
Gemini LLM 서비스 — google-genai SDK (신버전)
- 텍스트 + 이미지 멀티모달 지원
- 스트리밍 / 단발성 응답 모두 지원
- asyncio 기반 (동기 SDK를 thread pool로 래핑)
- 산책 말동무 전용 시스템 프롬프트
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ── 환경 변수 ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
TEMPERATURE = float(os.getenv("LLM_TEMP", "0.7"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ── 산책 말동무 시스템 프롬프트 ────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 치매 예방을 위한 산책 AI 말동무입니다.
어르신과 야외를 걸으며 자연스럽고 따뜻한 대화를 나눕니다.

[대화 방식]
- 친근한 존댓말, 2~3문장으로 짧게, 질문은 한 번에 하나만
- 어르신이 한 말에 공감하고 자연스럽게 이어받아 대화합니다
- 어르신이 꺼낸 주제(음식, 추억, 가족, 동네 이야기 등)를 자연스럽게 이어갑니다
- 같은 말을 반복해도 항상 처음 듣는 것처럼 따뜻하게 반응합니다
- 어르신이 힘들거나 불안해하면 먼저 공감하고 안심시킵니다
- 틀린 말도 지적하지 않고 자연스럽게 넘어갑니다

[하지 말 것]
- 정해진 주기로 억지로 사진을 요청하지 않습니다. 어르신이 아름다운 풍경이나 뭔가를 보고 있다고 말할 때 자연스럽게 한 번 물어볼 수 있습니다
- 특정 소재에 집착하지 않습니다. 대화 흐름을 자연스럽게 따라갑니다
- 이전 대화 내용을 요약하거나 다시 꺼내지 않습니다. 가장 최근 말에만 집중합니다
- 의학적 조언이나 위험한 내용은 말하지 않습니다

[사진을 받았을 때]
- 사진에 실제로 보이는 것만 묘사합니다. 불확실하면 "어떤 곳에서 찍으셨어요?"라고 여쭤봅니다
- 사진 묘사는 사진을 받은 그 응답 딱 1번만 합니다. 그 다음 메시지부터는 사진 내용을 절대 다시 꺼내지 않습니다
- 어르신이 사진에 대해 직접 말씀하셔도 짧게 호응하고 바로 새 주제로 전환합니다
- 현재 요청에 사진이 없으면 사진 관련 말은 일절 하지 않습니다"""


def _strip_photo_thanks(text: str) -> str:
    """AI 응답에서 사진 관련 내부 마커 및 감사 구문 제거."""
    import re
    text = re.sub(r'사진\s*보내\s*주셔서\s*감사합니다[!.!．]?\s*', '', text)
    text = re.sub(r'\[사진 설명 완료[^\]]*\]\s*', '', text)
    return text.strip()


def _build_contents(
    history: list[dict],
    user_text: str,
    image_bytes: Optional[bytes] = None,
    image_media_type: Optional[str] = None,
) -> list:
    """대화 히스토리 + 현재 입력을 google.genai contents 형식으로 변환."""
    contents = []

    recent = history[-20:] if len(history) > 20 else history
    prev_was_photo = False  # 직전 메시지가 [사진 전송]이었는지 추적
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            role = "model"
        # [사진 전송] / [사진 확인] — DB 저장용 placeholder, 명확한 과거형으로 교체
        if content in ("[사진 전송]", "[사진 확인]"):
            content = "[이전 대화에서 어르신이 사진을 보냈음. 현재 이 요청에는 사진 없음.]"
            prev_was_photo = True
        elif role == "model" and prev_was_photo:
            # 사진 직후 첫 번째 모델 응답 — '사진 보내주셔서 감사합니다' 제거 후
            # 사진 설명이 여기서 끝났음을 히스토리에 명시 → 이후 응답에서 재언급 차단
            content = _strip_photo_thanks(content)
            content += "\n[사진 설명 완료. 이후 대화에서 이 사진 내용을 다시 언급하지 않음.]"
            prev_was_photo = False
        else:
            # 모델 히스토리에서 '사진 보내주셔서 감사합니다' 항상 제거
            if role == "model":
                content = _strip_photo_thanks(content)
            prev_was_photo = False
        if role in ("user", "model") and content:
            contents.append(types.Content(role=role, parts=[types.Part(text=content)]))

    # 현재 사용자 입력
    parts = []
    if image_bytes:
        mime = image_media_type or "image/jpeg"
        parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=image_bytes)))
    if user_text:
        parts.append(types.Part(text=user_text))
    if not parts:
        parts.append(types.Part(text=""))

    contents.append(types.Content(role="user", parts=parts))
    return contents


def _get_time_context() -> str:
    """현재 KST 시간대 문자열 반환."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        import time as _time
        now = datetime.utcnow().replace(hour=(datetime.utcnow().hour + 9) % 24)
    hour = now.hour
    if 5 <= hour < 10:
        slot = "아침"
    elif 10 <= hour < 12:
        slot = "오전"
    elif 12 <= hour < 14:
        slot = "점심 시간"
    elif 14 <= hour < 18:
        slot = "오후"
    elif 18 <= hour < 21:
        slot = "저녁"
    else:
        slot = "밤"
    return f"{now.strftime('%Y년 %m월 %d일')} {slot} ({now.strftime('%H시 %M분')} KST)"


def _make_config(memory_context: str = "", has_image: bool = False) -> types.GenerateContentConfig:
    system = SYSTEM_PROMPT
    system += f"\n\n[현재 시간]\n{_get_time_context()}\n대화할 때 시간대(아침/오전/오후/저녁 등)를 정확히 반영하세요."
    if memory_context:
        system += f"\n\n[어르신 정보]\n{memory_context}"
    # 매 요청마다 이미지 유무를 명시적으로 주입 — LLM이 히스토리 패턴 따라 잘못 반응하는 것 방지
    if has_image:
        system += "\n\n[현재 요청] 이번 메시지에 사진이 첨부되어 있습니다. 사진에 실제로 보이는 것만 묘사하세요."
    else:
        system += (
            "\n\n[현재 요청] 이번 메시지에 사진이 없습니다. "
            "꽃, 나무, 풍경 등 보이지 않는 것을 상상해서 묘사하지 마세요. "
            "어르신이 하신 말씀에만 집중해서 답하세요."
        )
    return types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        # Gemini 2.5 Flash는 thinking을 기본 사용 — thinking 토큰이 max_output_tokens를
        # 잠식해 실제 응답이 잘림. 채팅 용도이므로 thinking 비활성화.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def _generate_sync(
    history: list[dict],
    user_text: str,
    memory_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_media_type: Optional[str] = None,
) -> str:
    """동기 생성 — thread pool에서 실행. 503/429 시 최대 3회 재시도."""
    import time
    from google.genai import errors as genai_errors

    client = _get_client()
    contents = _build_contents(history, user_text, image_bytes, image_media_type)
    config = _make_config(memory_context, has_image=bool(image_bytes))

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
            text = response.text or ""
            # '사진 보내주셔서 감사합니다'는 항상 제거
            text = _strip_photo_thanks(text)
            logger.info("Gemini response: finish_reason=%s, length=%d, text=%.100s", finish_reason, len(text), text)
            return text
        except genai_errors.ServerError as e:
            if attempt < 2:
                wait = (attempt + 1) * 3
                logger.warning("Gemini 503, %d초 후 재시도 (%d/3)...", wait, attempt + 1)
                time.sleep(wait)
            else:
                raise RuntimeError("AI 서버가 잠시 바쁩니다. 조금 후 다시 시도해 주세요.") from e
        except genai_errors.ClientError as e:
            raise RuntimeError(f"AI 요청 오류: {e}") from e
    return ""


async def stream_response(
    history: list[dict],
    user_text: str,
    memory_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_media_type: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """비동기 응답 제너레이터."""
    text = await generate_response(history, user_text, memory_context, image_bytes, image_media_type)
    if text:
        yield text


async def generate_response(
    history: list[dict],
    user_text: str,
    memory_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_media_type: Optional[str] = None,
) -> str:
    """완전한 응답 텍스트 반환."""
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(
        None, _generate_sync, history, user_text, memory_context, image_bytes, image_media_type
    )
    return text.strip()


async def warm_up_llm() -> None:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — LLM will fail at runtime")
        return
    logger.info("Gemini LLM ready (model=%s)", GEMINI_MODEL)
