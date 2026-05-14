"""
온디바이스 STT 서비스 — faster-whisper 기반
- 모델 싱글톤 (최초 1회 로드)
- sync inference를 asyncio thread pool로 래핑
- 한국어 특화 옵션 (language="ko", beam_size=5)
- VAD 필터로 무음 구간 자동 제거
"""

import asyncio
import io
import logging
import os
from functools import lru_cache
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ── 환경 변수 설정 ────────────────────────────────────────────────────────────
# WHISPER_MODEL: "tiny", "base", "small", "medium", "large-v3"
# 권장: "base" (속도/정확도 균형), 한국어는 "small" 이상 권장
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")      # "cpu" or "cuda"
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")   # "int8", "float16", "float32"


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
    """싱글톤 Whisper 모델 로드 (최초 호출 시 캐싱)."""
    logger.info(
        "Loading Whisper model: %s on %s (%s)",
        WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE,
    )
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        # faster-whisper는 기본적으로 ~/.cache/huggingface에 모델 다운로드
        # 오프라인 환경: download_root=os.getenv("WHISPER_CACHE_DIR", ".models/whisper")
        download_root=os.getenv("WHISPER_CACHE_DIR", None),
    )
    logger.info("Whisper model loaded successfully")
    return model


def _transcribe_sync(audio_bytes: bytes) -> tuple[str, float]:
    """
    동기 STT 처리 (thread pool에서 실행).
    Returns: (transcript_text, avg_logprob)
    """
    model = _get_whisper_model()

    audio_buffer = io.BytesIO(audio_bytes)

    segments, info = model.transcribe(
        audio_buffer,
        language="ko",
        beam_size=5,
        best_of=5,
        # VAD: 무음 구간 자동 필터링
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        # 반복 패턴 억제 (노인 발화에서 같은 단어 반복 시 루프 방지)
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    texts: list[str] = []
    total_logprob = 0.0
    seg_count = 0

    for seg in segments:
        text = seg.text.strip()
        if text:
            texts.append(text)
            total_logprob += seg.avg_logprob
            seg_count += 1

    transcript = " ".join(texts).strip()
    avg_logprob = total_logprob / seg_count if seg_count > 0 else -1.0

    logger.debug(
        "STT result: '%s' (lang=%s, prob=%.2f)",
        transcript, info.language, info.language_probability,
    )

    return transcript, avg_logprob


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.m4a",
) -> tuple[str, float]:
    """
    비동기 STT 엔트리포인트.
    Returns: (transcript_text, confidence_score)
    - confidence_score: avg_logprob (-inf~0), -0.5 이상이면 신뢰도 높음
    """
    if not audio_bytes:
        raise ValueError("오디오 데이터가 비어 있습니다.")

    loop = asyncio.get_event_loop()
    transcript, confidence = await loop.run_in_executor(
        None, _transcribe_sync, audio_bytes
    )

    if not transcript:
        logger.warning("STT returned empty transcript for file: %s", filename)
        return "", confidence

    return transcript, confidence


async def warm_up_stt() -> None:
    """
    서버 시작 시 모델 프리로드 (첫 요청 레이턴시 제거).
    무음 데이터로 더미 추론 실행.
    """
    try:
        logger.info("Warming up Whisper STT model...")
        # WAV 헤더만 있는 44바이트 더미 (무음)
        dummy_wav = bytes(44)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _transcribe_sync, dummy_wav)
        logger.info("STT warm-up complete")
    except Exception as e:
        # warm-up 실패는 치명적이지 않음 — 첫 요청 때 로드됨
        logger.warning("STT warm-up failed (will load on first request): %s", e)
