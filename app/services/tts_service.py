"""
온디바이스 TTS 서비스 — edge-tts 기반 (한국어 최적화)
- API 키 불필요, Microsoft Edge TTS 서비스 활용
- interruption handling: 요청 취소 지원
- 느린 발화 속도 (어르신 친화적 -15% 속도)
- 반환: MP3 바이트 (클라이언트에서 재생)

목소리:
- 여성(기본): ko-KR-SunHiNeural   — 따뜻하고 친근함
- 남성(선택): ko-KR-InJoonNeural  — 차분하고 안정적
"""

import asyncio
import io
import logging
import os
from typing import Optional

import edge_tts

logger = logging.getLogger(__name__)

# ── 환경 변수 ──────────────────────────────────────────────────────────────────
TTS_VOICE = os.getenv("TTS_VOICE", "ko-KR-SunHiNeural")
# 어르신 친화적: -15% 느린 속도, 조금 높은 피치로 명확하게
TTS_RATE = os.getenv("TTS_RATE", "-15%")
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")
TTS_VOLUME = os.getenv("TTS_VOLUME", "+10%")

# edge-tts 요청 타임아웃 (네트워크 불안정 대비)
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "15"))


async def synthesize(
    text: str,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
) -> bytes:
    """
    텍스트 → MP3 바이트 변환.

    Args:
        text: 합성할 텍스트 (빈 문자열이면 빈 bytes 반환)
        voice: 목소리 ID (None이면 환경변수 기본값 사용)
        rate: 속도 조절 (예: "-15%", "+0%")
        pitch: 피치 조절 (예: "+0Hz", "+5Hz")

    Returns:
        MP3 bytes (클라이언트에서 직접 재생 가능)
    """
    if not text or not text.strip():
        return b""

    _voice = voice or TTS_VOICE
    _rate = rate or TTS_RATE
    _pitch = pitch or TTS_PITCH

    # 어르신 배려: 괄호/이모지 제거, 문장 정리
    clean_text = _clean_text_for_tts(text)
    if not clean_text:
        return b""

    try:
        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=_voice,
            rate=_rate,
            pitch=_pitch,
            volume=TTS_VOLUME,
        )

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()

        if not audio_bytes:
            logger.warning("TTS returned empty audio for text: %.50s", clean_text)
            return b""

        logger.debug("TTS synthesized: %.50s (%d bytes)", clean_text, len(audio_bytes))
        return audio_bytes

    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)
        raise RuntimeError(f"TTS 합성 실패: {e}") from e


async def synthesize_with_timeout(
    text: str,
    timeout_sec: int = TTS_TIMEOUT,
    **kwargs,
) -> bytes:
    """타임아웃 포함 TTS (네트워크 불안정 환경용)."""
    try:
        return await asyncio.wait_for(synthesize(text, **kwargs), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.error("TTS timeout after %ds for text: %.50s", timeout_sec, text)
        raise RuntimeError("TTS 서비스 응답 시간 초과")


def _clean_text_for_tts(text: str) -> str:
    """
    TTS 전처리:
    - 마크다운 제거
    - 괄호 내용 제거
    - 연속 공백 정리
    - 과도한 문장 길이 트리밍 (300자 이상은 잘라냄)
    """
    import re

    # 마크다운 볼드/이탤릭
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # 괄호 주석 제거: (영어설명) 패턴
    text = re.sub(r"\([a-zA-Z\s]+\)", "", text)
    # URL 제거
    text = re.sub(r"https?://\S+", "", text)
    # 특수문자 정리 (한국어, 숫자, 기본 문장부호만 유지)
    text = re.sub(r"[^\w\s가-힣.,!?~\-]", " ", text)
    # 연속 공백 → 단일 공백
    text = re.sub(r"\s+", " ", text).strip()
    # 600자 이상이면 마지막 문장 경계에서 자름
    if len(text) > 600:
        cut = text[:600]
        last_punct = max(cut.rfind("."), cut.rfind("?"), cut.rfind("!"))
        text = cut[:last_punct + 1] if last_punct > 0 else cut

    return text


async def list_korean_voices() -> list[str]:
    """사용 가능한 한국어 목소리 목록 반환 (디버그용)."""
    voices = await edge_tts.list_voices()
    korean = [v["Name"] for v in voices if v["Locale"].startswith("ko-")]
    return korean
