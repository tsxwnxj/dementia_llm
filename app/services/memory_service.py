"""
메모리 서비스 — SQLite 기반 단기/장기/감정 기억
- short-term: DB에서 최근 N턴 메시지 로드
- long-term: 유저 요약 텍스트 (LLM 생성, 세션 종료 시 자동 업데이트)
- emotional: 반복 키워드/감정 추적
- topic_tracker: 반복 주제 감지 (같은 이야기 반복 패턴 파악)
"""

import asyncio
import logging
import time
import uuid
from typing import Optional

from app.db.database import get_db

logger = logging.getLogger(__name__)

# 단기 기억: DB에서 불러올 최근 메시지 수
SHORT_TERM_LIMIT = 20

# 감정 키워드 (긍정/부정 사전)
_POSITIVE_KEYWORDS = {
    "좋아", "행복", "기쁘", "즐거", "감사", "사랑", "예쁘", "맛있",
    "건강", "산책", "가족", "손자", "아이", "꽃", "봄", "따뜻",
}
_NEGATIVE_KEYWORDS = {
    "힘들", "아프", "무서", "슬프", "외로", "모르", "잊어", "걱정",
    "피곤", "어지", "무릎", "허리",
}


# ── 세션 ──────────────────────────────────────────────────────────────────────

async def create_session(uid: str) -> str:
    """새 산책 세션 생성, session_id 반환."""
    session_id = str(uuid.uuid4())
    async with get_db() as db:
        # 유저가 없으면 자동 생성
        await db.execute(
            "INSERT OR IGNORE INTO users(uid) VALUES (?)", (uid,)
        )
        await db.execute(
            "INSERT INTO sessions(session_id, uid) VALUES (?, ?)",
            (session_id, uid),
        )
        await db.commit()
    logger.info("Session created: %s for uid=%s", session_id, uid)
    return session_id


async def end_session(session_id: str) -> None:
    """세션 종료 시간 기록."""
    async with get_db() as db:
        await db.execute(
            "UPDATE sessions SET ended_at=? WHERE session_id=?",
            (int(time.time()), session_id),
        )
        await db.commit()


# ── 메시지 저장/로드 ──────────────────────────────────────────────────────────

async def save_message(
    session_id: str,
    uid: str,
    role: str,
    content: str,
    raw_stt: Optional[str] = None,
) -> None:
    """대화 메시지 저장 (role: 'user' | 'assistant')."""
    async with get_db() as db:
        # FK 위반 방지: user/session이 없으면 자동 생성
        await db.execute("INSERT OR IGNORE INTO users(uid) VALUES (?)", (uid,))
        await db.execute(
            "INSERT OR IGNORE INTO sessions(session_id, uid) VALUES (?, ?)",
            (session_id, uid),
        )
        await db.execute(
            """
            INSERT INTO messages(session_id, uid, role, content, raw_stt)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, uid, role, content, raw_stt),
        )
        await db.commit()

    # 유저 발화에서 감정 키워드 비동기 추적
    if role == "user":
        await _track_keywords(uid, content)


async def load_recent_messages(uid: str, limit: int = SHORT_TERM_LIMIT) -> list[dict]:
    """
    유저의 최근 N개 메시지 로드 (단기 기억).
    LLM 컨텍스트 히스토리로 사용.
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT role, content FROM messages
            WHERE uid=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (uid, limit),
        )
        rows = await cursor.fetchall()

    # DESC로 가져왔으니 역순으로 (오래된 것 → 최신)
    messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
    return messages


# ── 장기 기억 (유저 요약) ─────────────────────────────────────────────────────

async def get_long_term_summary(uid: str) -> str:
    """저장된 유저 장기 요약 반환 (없으면 빈 문자열)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT long_term_summary FROM users WHERE uid=?", (uid,)
        )
        row = await cursor.fetchone()
    return row["long_term_summary"] if row and row["long_term_summary"] else ""


async def update_long_term_summary(uid: str, summary: str) -> None:
    """LLM이 생성한 장기 요약 저장."""
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO users(uid, long_term_summary, updated_at)
            VALUES (?, ?, strftime('%s','now'))
            ON CONFLICT(uid) DO UPDATE SET
                long_term_summary=excluded.long_term_summary,
                updated_at=strftime('%s','now')
            """,
            (uid, summary),
        )
        await db.commit()
    logger.info("Long-term summary updated for uid=%s", uid)


async def build_memory_context(uid: str) -> str:
    """
    LLM 시스템 프롬프트에 삽입할 메모리 컨텍스트 구성.
    - 장기 요약
    - 자주 언급한 주제 TOP 5
    - 최근 감정 상태
    """
    summary = await get_long_term_summary(uid)
    top_topics = await get_top_topics(uid, limit=5)
    recent_emotions = await get_recent_emotions(uid, limit=3)

    parts: list[str] = []
    if summary:
        parts.append(f"어르신 정보: {summary}")
    if top_topics:
        topics_str = ", ".join(top_topics)
        parts.append(f"자주 하시는 이야기: {topics_str}")
    if recent_emotions:
        emotions_str = ", ".join(recent_emotions)
        parts.append(f"최근 자주 말씀하신 것들: {emotions_str}")

    return "\n".join(parts)


# ── 감정/주제 추적 ────────────────────────────────────────────────────────────

async def _track_keywords(uid: str, text: str) -> None:
    """유저 발화에서 감정 키워드 추출 후 DB 저장."""
    found_positive = [kw for kw in _POSITIVE_KEYWORDS if kw in text]
    found_negative = [kw for kw in _NEGATIVE_KEYWORDS if kw in text]

    if not found_positive and not found_negative:
        return

    async with get_db() as db:
        for kw in found_positive:
            await db.execute(
                """
                INSERT INTO emotional_memory(uid, keyword, emotion, mention_count, last_mentioned)
                VALUES (?, ?, 'positive', 1, strftime('%s','now'))
                ON CONFLICT DO NOTHING
                """,
                (uid, kw),
            )
            await db.execute(
                """
                UPDATE emotional_memory
                SET mention_count=mention_count+1, last_mentioned=strftime('%s','now')
                WHERE uid=? AND keyword=?
                """,
                (uid, kw),
            )
        for kw in found_negative:
            await db.execute(
                """
                INSERT INTO emotional_memory(uid, keyword, emotion, mention_count, last_mentioned)
                VALUES (?, ?, 'negative', 1, strftime('%s','now'))
                ON CONFLICT DO NOTHING
                """,
                (uid, kw),
            )
            await db.execute(
                """
                UPDATE emotional_memory
                SET mention_count=mention_count+1, last_mentioned=strftime('%s','now')
                WHERE uid=? AND keyword=?
                """,
                (uid, kw),
            )
        await db.commit()


async def track_topic(uid: str, topic: str) -> None:
    """반복 주제 기록 (LLM이 추출한 주제 키워드)."""
    if not topic or not topic.strip():
        return
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO topic_tracker(uid, topic, count, last_at)
            VALUES (?, ?, 1, strftime('%s','now'))
            ON CONFLICT(uid, topic) DO UPDATE SET
                count=count+1,
                last_at=strftime('%s','now')
            """,
            (uid, topic.strip()),
        )
        await db.commit()


async def get_top_topics(uid: str, limit: int = 5) -> list[str]:
    """가장 많이 언급된 주제 반환."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT topic FROM topic_tracker
            WHERE uid=?
            ORDER BY count DESC, last_at DESC
            LIMIT ?
            """,
            (uid, limit),
        )
        rows = await cursor.fetchall()
    return [row["topic"] for row in rows]


async def get_recent_emotions(uid: str, limit: int = 3) -> list[str]:
    """최근에 많이 언급된 감정 키워드 반환."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT keyword FROM emotional_memory
            WHERE uid=?
            ORDER BY mention_count DESC, last_mentioned DESC
            LIMIT ?
            """,
            (uid, limit),
        )
        rows = await cursor.fetchall()
    return [row["keyword"] for row in rows]


# ── 장기 기억 자동 요약 ───────────────────────────────────────────────────────

# 요약 트리거 최소 유저 발화 수 (너무 짧은 세션은 요약 안 함)
_MIN_MESSAGES_TO_SUMMARIZE = 3

_SUMMARY_PROMPT_TEMPLATE = """\
아래는 산책 AI와 어르신의 대화 기록입니다.
기존 어르신 정보와 새 대화를 합쳐서, 어르신에 대해 알게 된 사실을 간결하게 정리해 주세요.

[기존 어르신 정보]
{existing_summary}

[오늘 대화 ({date})]
{conversation}

[정리 규칙]
- 어르신의 이름/호칭, 가족 관계, 좋아하는 것, 건강 상태, 고향/추억, 반복 주제만 포함
- 새로 알게 된 사실이 있으면 추가, 기존 내용과 중복되면 합치기
- 3~8줄 이내 bullet 형식 (예: • 손자 이름: 민준)
- 없는 정보는 추가하지 말 것
- 사진, 이미지, 풍경 묘사와 관련된 내용은 절대 포함하지 말 것
- 한국어로 작성

[정리된 어르신 정보]"""


async def get_session_messages(session_id: str) -> list[dict]:
    """특정 세션의 전체 대화 메시지 반환."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id=?
            ORDER BY id ASC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


async def summarize_and_update(uid: str, session_id: str) -> str:
    """
    세션 대화를 LLM으로 요약해 장기 기억에 누적 저장.
    백그라운드로 실행 — UI를 블록하지 않음.

    Returns: 생성된 요약 텍스트 (빈 문자열이면 건너뜀)
    """
    from app.services import llm_service  # 순환 import 방지

    # 세션 메시지 로드
    messages = await get_session_messages(session_id)
    user_turns = [m for m in messages if m["role"] == "user"]

    if len(user_turns) < _MIN_MESSAGES_TO_SUMMARIZE:
        logger.info(
            "Session %s skipped summarization (only %d user turns)",
            session_id, len(user_turns),
        )
        return ""

    # 대화를 텍스트로 변환 — 사진 전송 메시지와 직후 AI 사진 묘사 응답 제외
    conv_lines = []
    skip_next_ai = False
    for m in messages:
        content = m["content"]
        role = m["role"]
        # 사진 관련 메시지 → 모두 건너뜀 (기억 저장 완전 차단)
        if content in ("[사진 전송]", "[사진 확인]") or content.startswith("[이전 대화에서 어르신이 사진을"):
            skip_next_ai = True
            continue
        if role == "assistant" and skip_next_ai:
            skip_next_ai = False
            continue
        skip_next_ai = False
        conv_lines.append(f"{'어르신' if role == 'user' else 'AI'}: {content}")
    conversation_text = "\n".join(conv_lines)

    # 기존 장기 요약 로드
    existing_summary = await get_long_term_summary(uid)
    if not existing_summary:
        existing_summary = "(아직 없음)"

    from datetime import date
    today = date.today().strftime("%Y년 %m월 %d일")

    prompt = _SUMMARY_PROMPT_TEMPLATE.format(
        existing_summary=existing_summary,
        date=today,
        conversation=conversation_text,
    )

    try:
        new_summary = await llm_service.generate_response(
            history=[],
            user_text=prompt,
            memory_context="",
        )
        new_summary = new_summary.strip()

        if new_summary:
            await update_long_term_summary(uid, new_summary)
            logger.info(
                "Long-term summary updated for uid=%s (session=%s, %d turns)",
                uid, session_id, len(user_turns),
            )

        return new_summary

    except Exception as e:
        logger.error("Failed to summarize session %s: %s", session_id, e)
        return ""


def schedule_summarization(uid: str, session_id: str) -> None:
    """
    요약을 백그라운드 태스크로 예약.
    /walk/end 엔드포인트에서 호출 — 응답을 블록하지 않음.
    """
    asyncio.create_task(
        summarize_and_update(uid, session_id),
        name=f"summarize-{session_id[:8]}",
    )
    logger.info("Summarization scheduled for session=%s uid=%s", session_id, uid)


# ── 개인화 발음 사전 ──────────────────────────────────────────────────────────

async def get_user_pronunciation_dict(uid: str) -> dict[str, str]:
    """유저별 개인화 발음 매핑 {wrong: correct} 반환."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT wrong, correct FROM pronunciation_dict WHERE uid=? ORDER BY count DESC",
            (uid,),
        )
        rows = await cursor.fetchall()
    return {row["wrong"]: row["correct"] for row in rows}


async def record_pronunciation_correction(
    uid: str, wrong: str, correct: str
) -> None:
    """발음 보정 사례 기록 (누적 학습용)."""
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO pronunciation_dict(uid, wrong, correct, count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(uid, wrong) DO UPDATE SET
                correct=excluded.correct,
                count=count+1
            """,
            (uid, wrong, correct),
        )
        await db.commit()
