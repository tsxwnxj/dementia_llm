"""
STT-LLM-TTS 산책 AI 서비스
- STT: OpenAI Whisper (한국어 음성 → 텍스트)
- LLM: Google Gemini 2.5 Flash (google-genai 패키지)
- 날씨: Open-Meteo API (무료, API 키 불필요)
"""

import io
import os
from typing import Optional

import httpx
from google import genai
from google.genai import types
from openai import AsyncOpenAI

# ─── 클라이언트 초기화 ────────────────────────────────────────────────────────

gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gemini-2.5-flash"

# ─── 날씨 코드 매핑 (WMO) ────────────────────────────────────────────────────

WMO_WEATHER = {
    0: "맑음", 1: "대체로 맑음", 2: "부분적으로 흐림", 3: "흐림",
    45: "안개", 48: "안개",
    51: "가벼운 이슬비", 53: "이슬비", 55: "강한 이슬비",
    61: "가벼운 비", 63: "비", 65: "강한 비",
    71: "가벼운 눈", 73: "눈", 75: "강한 눈",
    80: "소나기", 81: "소나기", 82: "강한 소나기",
    95: "뇌우",
}

# ─── 시스템 프롬프트 ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 치매 예방을 위한 산책 AI 동반자입니다.
어르신과 함께 산책하며 따뜻한 대화로 인지 기능을 활성화해 드립니다.

[대화 원칙]
- 따뜻하고 친근한 존댓말로 대화합니다
- 2~3문장으로 짧게 말하고 반드시 질문 하나로 마칩니다
- 주변 환경(꽃, 나무, 하늘, 건물), 날씨, 계절에 대한 질문을 합니다
- 어릴 적 추억, 좋아하는 것, 감정을 불러일으키는 질문을 합니다
- 색깔, 모양, 개수를 묻는 인지 훈련 질문을 가끔 합니다
- 사진이 첨부되면 사진 속 내용을 친절하게 묘사하고 관련 질문을 합니다
- 절대 의학적 조언이나 위험한 내용은 말하지 않습니다"""

GEMINI_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    max_output_tokens=300,
)

GEMINI_CONFIG_SIMPLE = types.GenerateContentConfig(
    max_output_tokens=200,
)

# ─── 날씨 ────────────────────────────────────────────────────────────────────

def _uv_info(uv: float) -> tuple[str, str]:
    if uv < 3:    return "낮음", ""
    elif uv < 6:  return "보통", "자외선 차단제를 바르는 것을 추천드려요"
    elif uv < 8:  return "높음", "자외선 차단제와 모자를 챙기세요"
    elif uv < 11: return "매우 높음", "양산이나 모자를 꼭 챙기세요"
    else:         return "위험", "가능하면 실내에 계시는 게 좋을 것 같아요"


async def fetch_weather(lat: float, lon: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,uv_index,weathercode",
                    "timezone": "Asia/Seoul",
                },
            )
            if resp.status_code == 200:
                current = resp.json().get("current", {})
                return {
                    "temperature": round(current.get("temperature_2m", 20)),
                    "uv_index": current.get("uv_index", 3.0),
                    "weather_code": int(current.get("weathercode", 0)),
                }
    except Exception:
        pass
    return {"temperature": 20, "uv_index": 3.0, "weather_code": 0}


async def generate_walk_greeting(lat: Optional[float], lon: Optional[float]) -> str:
    weather_text = "맑은 날씨"
    uv_advice = ""

    if lat is not None and lon is not None:
        try:
            w = await fetch_weather(lat, lon)
            weather_desc = WMO_WEATHER.get(w["weather_code"], "맑음")
            uv_level, uv_adv = _uv_info(w["uv_index"])
            weather_text = f"{weather_desc}, 기온 {w['temperature']}도, 자외선 {uv_level}"
            uv_advice = f" {uv_adv}." if uv_adv else "."
        except Exception:
            pass

    prompt = (
        f"오늘 산책을 막 시작한 어르신에게 날씨 정보를 알려드리고 산책을 응원하는 짧은 인사말을 만들어주세요.\n"
        f"날씨: {weather_text}{uv_advice}\n"
        f"조건: 2~3문장, 따뜻한 존댓말, 마지막에 주변 환경에 대한 질문 하나 포함."
    )

    response = await gemini.aio.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=GEMINI_CONFIG_SIMPLE,
    )
    return response.text.strip()


async def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    transcript = await openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ko",
    )
    return transcript.text.strip()


async def process_chat(
    messages: list[dict],
    user_text: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_media_type: Optional[str] = None,
) -> str:
    contents: list[types.Content] = []
    for msg in messages[-20:]:
        role = "model" if msg.get("role") == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )

    current_parts: list[types.Part] = []
    if image_bytes:
        current_parts.append(
            types.Part.from_bytes(data=image_bytes, mime_type=image_media_type or "image/jpeg")
        )
    current_parts.append(
        types.Part.from_text(text=user_text if user_text else "이 사진을 보세요.")
    )
    contents.append(types.Content(role="user", parts=current_parts))

    response = await gemini.aio.models.generate_content(
        model=MODEL,
        contents=contents,
        config=GEMINI_CONFIG,
    )
    return response.text.strip()
