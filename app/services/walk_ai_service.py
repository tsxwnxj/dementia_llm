"""
산책 AI 서비스 — 온디바이스 파이프라인 통합
기존 함수 시그니처 유지, 내부 구현을 로컬 서비스로 교체.

Pipeline:
  Audio → [STT] → [발음 정규화] → [Memory 컨텍스트] → [LLM] → [TTS]
                                                              ↓
                                                        텍스트 + 오디오 반환

공개 함수 (기존 routers와 호환):
  - generate_walk_greeting(lat, lon)         → str
  - transcribe_audio(audio_bytes, filename)  → str  (STT만)
  - process_chat(messages, user_text, ...)   → str  (LLM만)

신규 함수 (확장):
  - process_chat_full(...)  → (ai_text, audio_bytes, corrections)
  - stream_chat(...)        → AsyncGenerator[str]
"""

import logging
from collections.abc import AsyncGenerator
from typing import Optional

import httpx

from app.services import llm_service, memory_service, pronunciation_service, stt_service, tts_service

logger = logging.getLogger(__name__)

# ── 날씨 코드 매핑 (WMO) — 기존 유지 ─────────────────────────────────────────

WMO_WEATHER = {
    0: "맑음", 1: "대체로 맑음", 2: "부분적으로 흐림", 3: "흐림",
    45: "안개", 48: "안개",
    51: "가벼운 이슬비", 53: "이슬비", 55: "강한 이슬비",
    61: "가벼운 비", 63: "비", 65: "강한 비",
    71: "가벼운 눈", 73: "눈", 75: "강한 눈",
    80: "소나기", 81: "소나기", 82: "강한 소나기",
    95: "뇌우",
}


def _uv_info(uv: float) -> tuple[str, str]:
    if uv < 3:    return "낮음", ""
    elif uv < 6:  return "보통", "자외선 차단제를 바르는 것을 추천드려요"
    elif uv < 8:  return "높음", "자외선 차단제와 모자를 챙기세요"
    elif uv < 11: return "매우 높음", "양산이나 모자를 꼭 챙기세요"
    else:         return "위험", "가능하면 실내에 계시는 게 좋을 것 같아요"


async def fetch_weather(lat: float, lon: float) -> dict:
    """Open-Meteo 무료 API로 현재 날씨 조회."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,relativehumidity_2m,uv_index,weathercode,windspeed_10m,precipitation",
                    "timezone": "Asia/Seoul",
                },
            )
            if resp.status_code == 200:
                current = resp.json().get("current", {})
                return {
                    "temperature": round(current.get("temperature_2m", 20)),
                    "humidity": round(current.get("relativehumidity_2m", 50)),
                    "uv_index": current.get("uv_index", 3.0),
                    "weather_code": int(current.get("weathercode", 0)),
                    "windspeed": current.get("windspeed_10m", 0.0),
                    "precipitation": current.get("precipitation", 0.0),
                }
    except Exception:
        pass
    return {"temperature": 20, "humidity": 50, "uv_index": 3.0, "weather_code": 0, "windspeed": 0.0, "precipitation": 0.0}


def _prep_checklist(w: dict) -> list[str]:
    """날씨 조건에 따른 외출 준비물/주의사항 목록 반환."""
    items = []
    code = w["weather_code"]
    temp = w["temperature"]
    uv = w["uv_index"]
    wind = w["windspeed"]
    precip = w["precipitation"]

    # 강수
    if code in (65, 75, 82):  # 강한 비/눈/소나기
        items.append("비가 많이 오니 우산을 꼭 챙기세요")
    elif code in (51, 53, 55, 61, 63, 71, 73, 80, 81):  # 이슬비/비/눈
        items.append("우산을 챙겨 나가시는 게 좋겠어요")
    elif precip > 0:
        items.append("비가 살짝 올 수 있으니 우산을 챙기시면 좋겠어요")

    # 자외선
    if uv >= 11:
        items.append("자외선이 매우 강하니 선크림, 양산, 모자를 꼭 챙기세요")
    elif uv >= 8:
        items.append("자외선이 강하니 양산이나 모자를 챙기세요")
    elif uv >= 6:
        items.append("햇빛이 꽤 강하니 선크림을 바르고 나가시면 좋겠어요")

    # 기온
    if temp >= 30:
        items.append("날씨가 많이 더우니 물을 꼭 챙기시고 짧은 산책을 권장드려요")
    elif temp >= 27:
        items.append("더운 날씨니 물을 챙겨 가세요")
    elif temp <= 5:
        items.append("날씨가 많이 추우니 두꺼운 외투를 입고 나가세요")
    elif temp <= 10:
        items.append("쌀쌀하니 따뜻한 겉옷을 챙기세요")

    # 강풍
    if wind >= 40:
        items.append("바람이 매우 강하니 외출을 삼가시거나 바람막이를 입으세요")
    elif wind >= 25:
        items.append("바람이 강하게 부니 바람막이 옷을 챙기세요")

    return items


# ── 산책 시작 인사말 (기존 시그니처 유지) ────────────────────────────────────

import random

_QUESTIONS_BY_SLOT = {
    "아침":  [
        "오늘 아침 산책길은 어떤가요?",
        "아침 공기가 어떻게 느껴지시나요?",
        "주변에 어떤 풍경이 눈에 들어오시나요?",
    ],
    "오전":  [
        "오전 산책 기분이 어떠세요?",
        "주변에 어떤 것들이 보이시나요?",
        "오늘 어떤 길을 걸으실 계획이세요?",
    ],
    "점심":  [
        "점심 드시고 나오셨나요?",
        "주변에 어떤 풍경이 펼쳐져 있나요?",
        "오늘 걷기 좋은 곳으로 나오셨네요, 어디를 걸으실 건가요?",
    ],
    "오후":  [
        "오후 산책 기분이 어떠세요?",
        "주변에 어떤 것들이 눈에 들어오시나요?",
        "오늘 걸으시는 곳이 어딘가요?",
    ],
    "저녁":  [
        "저녁 산책 나오신 건가요?",
        "저녁 공기가 어떻게 느껴지시나요?",
        "지금 걸으시는 곳 주변에 어떤 풍경이 보이시나요?",
    ],
    "밤":    [
        "밤 산책 나오셨네요, 어디를 걸으실 건가요?",
        "지금 걸으시는 곳 주변은 어떤가요?",
    ],
}


def _time_slot() -> tuple[str, str]:
    """현재 KST 시간대 (slot, greeting_phrase) 반환."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        now = datetime.utcnow()
    hour = now.hour
    if 5 <= hour < 10:
        return "아침", "좋은 아침이에요!"
    elif 10 <= hour < 12:
        return "오전", "안녕하세요!"
    elif 12 <= hour < 14:
        return "점심", "점심 시간에 산책이시네요!"
    elif 14 <= hour < 18:
        return "오후", "안녕하세요!"
    elif 18 <= hour < 21:
        return "저녁", "좋은 저녁이에요!"
    else:
        return "밤", "안녕하세요!"


async def generate_walk_greeting(
    lat: Optional[float],
    lon: Optional[float],
    uid: Optional[str] = None,
) -> str:
    """날씨 브리핑 + 준비사항을 Python 템플릿으로 100% 고정 생성 (LLM 미사용)."""
    slot, time_greeting = _time_slot()

    # ── 1. 날씨 조회 ─────────────────────────────────────────────────────────
    weather_desc = None
    temp: Optional[int] = None
    humidity: Optional[int] = None
    uv: Optional[float] = None
    prep_items: list[str] = []

    if lat is not None and lon is not None:
        try:
            w = await fetch_weather(lat, lon)
            weather_desc = WMO_WEATHER.get(w["weather_code"], "맑음")
            temp = w["temperature"]
            humidity = w.get("humidity")
            uv = w.get("uv_index")
            prep_items = _prep_checklist(w)
        except Exception as e:
            logger.warning("Weather fetch failed: %s", e)

    # ── 2. 날씨 브리핑 문장 조립 ─────────────────────────────────────────────
    # WMO 코드는 명사형("맑음","비" 등)이라 직접 쓰면 문법 오류 → 형태 변환 필요
    _WEATHER_ADJ = {
        "맑음":            "맑고",
        "대체로 맑음":      "대체로 맑고",
        "부분적으로 흐림":  "조금 흐리고",
        "흐림":            "흐리고",
        "안개":            "안개가 끼어 있고",
        "가벼운 이슬비":    "이슬비가 살짝 내리고",
        "이슬비":          "이슬비가 내리고",
        "강한 이슬비":      "이슬비가 꽤 내리고",
        "가벼운 비":        "비가 살짝 오고",
        "비":              "비가 오고",
        "강한 비":          "비가 많이 오고",
        "가벼운 눈":        "눈이 살짝 내리고",
        "눈":              "눈이 내리고",
        "강한 눈":          "눈이 많이 내리고",
        "소나기":          "소나기가 내리고",
        "강한 소나기":      "소나기가 많이 내리고",
        "뇌우":            "뇌우가 치고",
    }
    if weather_desc:
        conj = _WEATHER_ADJ.get(weather_desc, f"{weather_desc}이고")
        # 수치 브리핑: 날씨 상태 + 기온 + 습도 + 자외선
        parts_w = [f"날씨는 {conj}"]
        if temp is not None:
            parts_w.append(f"기온 {temp}도")
        if humidity is not None:
            parts_w.append(f"습도 {humidity}%")
        if uv is not None:
            uv_level, _ = _uv_info(uv)
            parts_w.append(f"자외선 지수 {round(uv)}({uv_level})")
        weather_sentence = "오늘 " + ", ".join(parts_w) + "예요."
    else:
        weather_sentence = "오늘 날씨 정보를 가져오지 못했어요."

    # ── 3. 준비사항 문장 조립 ─────────────────────────────────────────────────
    prep_sentence = " ".join(prep_items) if prep_items else ""

    # ── 4. 마무리 질문 (랜덤 템플릿) ─────────────────────────────────────────
    question = random.choice(_QUESTIONS_BY_SLOT.get(slot, _QUESTIONS_BY_SLOT["오후"]))

    # ── 5. 최종 조립 ─────────────────────────────────────────────────────────
    parts = [time_greeting, weather_sentence]
    if prep_sentence:
        parts.append(prep_sentence)
    parts.append(question)

    greeting = " ".join(parts)
    logger.info("Walk greeting: %s", greeting)
    return greeting


# ── STT (기존 시그니처 유지) ──────────────────────────────────────────────────

async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.m4a",
    uid: Optional[str] = None,
) -> str:
    """
    음성 → 텍스트 (온디바이스 faster-whisper).
    기존: OpenAI Whisper API → 변경: local faster-whisper
    발음 정규화 없이 raw STT만 반환 (process_chat_full에서 정규화 적용).
    """
    transcript, confidence = await stt_service.transcribe_audio(audio_bytes, filename)

    if confidence < -1.5:
        logger.warning(
            "Low STT confidence (%.2f) for uid=%s, file=%s",
            confidence, uid, filename,
        )

    return transcript


# ── LLM 대화 (기존 시그니처 유지) ────────────────────────────────────────────

async def process_chat(
    messages: list[dict],
    user_text: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_media_type: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    """텍스트/이미지 → LLM 응답 텍스트 (Gemini Flash 2.5 멀티모달)."""
    if not user_text and not image_bytes:
        raise ValueError("user_text 또는 image가 필요합니다.")

    # 메모리 컨텍스트
    memory_context = ""
    if uid:
        try:
            memory_context = await memory_service.build_memory_context(uid)
        except Exception as e:
            logger.warning("Failed to load memory context: %s", e)

    ai_text = await llm_service.generate_response(
        history=messages,
        user_text=user_text or "",
        memory_context=memory_context,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
    )
    return ai_text


# ── 풀 파이프라인 (신규 — 라우터에서 사용) ────────────────────────────────────

async def process_chat_full(
    session_id: str,
    uid: str,
    messages: list[dict],
    audio_bytes: Optional[bytes] = None,
    audio_filename: str = "audio.m4a",
    text_input: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_media_type: Optional[str] = None,
    include_tts: bool = True,
) -> dict:
    """
    STT → 발음정규화 → LLM → TTS 풀 파이프라인.

    Returns:
        {
            "user_text": str,          # 정규화된 사용자 입력
            "raw_stt": str | None,     # STT 원본 (발음 보정 전)
            "ai_text": str,            # LLM 응답
            "audio_bytes": bytes,      # TTS 오디오 (MP3), include_tts=False면 b""
            "corrections": list[dict], # 발음 보정 목록
        }
    """
    raw_stt: Optional[str] = None
    user_text: Optional[str] = text_input
    corrections: list[dict] = []

    # ── 1. STT ────────────────────────────────────────────────────────────────
    if audio_bytes and not user_text:
        raw_stt, _confidence = await stt_service.transcribe_audio(audio_bytes, audio_filename)
        # 개인화 발음 사전 로드
        user_dict = await memory_service.get_user_pronunciation_dict(uid)
        # 발음 정규화
        user_text, corrections = pronunciation_service.normalize(raw_stt, user_dict)

        # 보정 사례 학습 저장
        for c in corrections:
            await memory_service.record_pronunciation_correction(
                uid, c["original"], c["corrected"]
            )

        logger.info(
            "STT: '%s' → normalized: '%s' (%d corrections)",
            raw_stt, user_text, len(corrections),
        )

    if not user_text and not image_bytes:
        raise ValueError("텍스트, 음성, 이미지 중 하나 이상이 필요합니다.")

    # ── 2. 메모리 컨텍스트 ────────────────────────────────────────────────────
    memory_context = await memory_service.build_memory_context(uid)

    # 현재 세션 메시지만 히스토리로 사용 (이전 세션 오염 방지)
    db_history = await memory_service.get_session_messages(session_id)
    # 파라미터로 전달된 messages가 더 최신일 수 있으므로 DB와 병합
    combined_history = _merge_histories(db_history, messages)

    # ── 3. LLM 응답 생성 ──────────────────────────────────────────────────────
    ai_text = await llm_service.generate_response(
        history=combined_history,
        user_text=user_text or "",
        memory_context=memory_context,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
    )

    # ── 4. 메모리 저장 ────────────────────────────────────────────────────────
    # 이미지만 보낸 경우 "[사진 전송]"으로 저장
    saved_user_content = user_text or ("[사진 전송]" if image_bytes else "")
    await memory_service.save_message(
        session_id=session_id,
        uid=uid,
        role="user",
        content=saved_user_content,
        raw_stt=raw_stt,
    )
    # 사진 응답은 실제 묘사 텍스트 대신 플레이스홀더로 저장 → 장기 기억에 사진 내용 남지 않음
    saved_ai_content = "[사진 확인]" if image_bytes else ai_text
    await memory_service.save_message(
        session_id=session_id,
        uid=uid,
        role="assistant",
        content=saved_ai_content,
    )

    # ── 5. TTS 합성 ───────────────────────────────────────────────────────────
    audio_bytes_out = b""
    if include_tts and ai_text:
        try:
            audio_bytes_out = await tts_service.synthesize_with_timeout(ai_text)
        except Exception as e:
            logger.error("TTS failed, returning text only: %s", e)

    return {
        "user_text": user_text,
        "raw_stt": raw_stt,
        "ai_text": ai_text,
        "audio_bytes": audio_bytes_out,
        "corrections": corrections,
    }


async def stream_chat(
    session_id: str,
    uid: str,
    messages: list[dict],
    user_text: str,
) -> AsyncGenerator[str, None]:
    """
    LLM 응답 스트리밍 제너레이터 (SSE 엔드포인트용).
    TTS 없이 토큰만 스트리밍.
    """
    memory_context = await memory_service.build_memory_context(uid)
    db_history = await memory_service.load_recent_messages(uid)
    combined_history = _merge_histories(db_history, messages)

    full_response: list[str] = []

    async for token in llm_service.stream_response(
        history=combined_history,
        user_text=user_text,
        memory_context=memory_context,
    ):
        full_response.append(token)
        yield token

    # 완성된 응답을 메모리에 저장
    ai_text = "".join(full_response)
    if ai_text:
        await memory_service.save_message(session_id, uid, "user", user_text)
        await memory_service.save_message(session_id, uid, "assistant", ai_text)


# ── 세션 종료 + 장기 기억 요약 ────────────────────────────────────────────────

async def end_session(session_id: str, uid: str) -> None:
    """
    산책 세션 종료 처리.
    1. DB에 종료 시간 기록
    2. 세션 대화를 백그라운드에서 LLM으로 요약 → long_term_summary 누적
    """
    await memory_service.end_session(session_id)
    memory_service.schedule_summarization(uid, session_id)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _merge_histories(db_history: list[dict], param_history: list[dict]) -> list[dict]:
    """
    DB 히스토리 + 파라미터 히스토리 병합.
    중복 제거: content 기준으로 마지막 20턴만 유지.
    """
    seen: set[str] = set()
    merged: list[dict] = []

    for msg in db_history + param_history:
        key = f"{msg.get('role')}:{msg.get('content', '')[:50]}"
        if key not in seen:
            seen.add(key)
            merged.append(msg)

    return merged[-20:]
