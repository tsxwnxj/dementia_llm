"""
SQLite async database layer.
- aiosqlite 기반
- 앱 시작 시 테이블 자동 생성
- 전역 connection pool (단일 파일 DB)
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/dementia_app.db")

# ── 테이블 DDL ────────────────────────────────────────────────────────────────

_DDL = """
-- 유저별 메타 정보 (장기 기억 요약 포함)
CREATE TABLE IF NOT EXISTS users (
    uid             TEXT PRIMARY KEY,
    display_name    TEXT,
    long_term_summary TEXT DEFAULT '',   -- LLM이 생성한 유저 요약
    created_at      INTEGER DEFAULT (strftime('%s','now')),
    updated_at      INTEGER DEFAULT (strftime('%s','now'))
);

-- 산책 세션
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    uid             TEXT NOT NULL,
    started_at      INTEGER DEFAULT (strftime('%s','now')),
    ended_at        INTEGER,
    FOREIGN KEY (uid) REFERENCES users(uid)
);

-- 대화 메시지 (단기 기억 원본)
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    uid             TEXT NOT NULL,
    role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content         TEXT NOT NULL,
    raw_stt         TEXT,           -- STT 원본 (발음 보정 전)
    created_at      INTEGER DEFAULT (strftime('%s','now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- 감정 기억 (대화에서 감지된 감정/키워드)
CREATE TABLE IF NOT EXISTS emotional_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             TEXT NOT NULL,
    keyword         TEXT NOT NULL,  -- '강아지', '손자', '고향'
    emotion         TEXT,           -- 'positive', 'negative', 'neutral'
    mention_count   INTEGER DEFAULT 1,
    last_mentioned  INTEGER DEFAULT (strftime('%s','now')),
    FOREIGN KEY (uid) REFERENCES users(uid)
);

-- 반복 주제 추적
CREATE TABLE IF NOT EXISTS topic_tracker (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             TEXT NOT NULL,
    topic           TEXT NOT NULL,
    count           INTEGER DEFAULT 1,
    last_at         INTEGER DEFAULT (strftime('%s','now')),
    UNIQUE(uid, topic),
    FOREIGN KEY (uid) REFERENCES users(uid)
);

-- 개인화 발음 사전 (유저별 사투리/발음 매핑)
CREATE TABLE IF NOT EXISTS pronunciation_dict (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uid             TEXT NOT NULL,
    wrong           TEXT NOT NULL,  -- 인식된 텍스트
    correct         TEXT NOT NULL,  -- 올바른 텍스트
    count           INTEGER DEFAULT 1,
    UNIQUE(uid, wrong),
    FOREIGN KEY (uid) REFERENCES users(uid)
);

CREATE INDEX IF NOT EXISTS idx_messages_uid ON messages(uid);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_emotional_uid ON emotional_memory(uid);
CREATE INDEX IF NOT EXISTS idx_topic_uid ON topic_tracker(uid);
"""


async def init_db() -> None:
    """DB 파일 생성 및 스키마 초기화."""
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_DDL)
        await db.commit()
    logger.info("DB initialized: %s", DB_PATH)


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """요청당 DB 커넥션 컨텍스트 매니저."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        yield db
