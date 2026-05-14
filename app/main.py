from dotenv import load_dotenv
load_dotenv()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.routers import health, motion, walk_chat
import firebase_admin
from firebase_admin import credentials
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Firebase Admin 초기화 ─────────────────────────────────────────────────────
if os.path.isfile("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)


# ── 앱 수명주기: DB 초기화 + 모델 프리로드 ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 시작 시 ───────────────────────────────────────────────────────────────
    logger.info("Starting DementiaApp API server...")

    # 1. SQLite DB 초기화 (테이블 생성)
    from app.db.database import init_db
    await init_db()

    # 2. 모델 프리로드 (첫 요청 레이턴시 제거)
    from app.services.stt_service import warm_up_stt
    from app.services.llm_service import warm_up_llm

    import asyncio
    # STT와 LLM 프리로드 병렬 실행
    await asyncio.gather(
        warm_up_stt(),
        warm_up_llm(),
        return_exceptions=True,  # 하나 실패해도 서버 시작 계속
    )

    logger.info("Server startup complete")
    yield

    # ── 종료 시 ───────────────────────────────────────────────────────────────
    logger.info("Server shutting down...")


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="DementiaApp API",
    version="2.0.0",
    description="온디바이스 AI 기반 치매 예방 산책 동반자",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 수정
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health.router)
app.include_router(motion.router, prefix="/api/v1")
app.include_router(walk_chat.router, prefix="/api/v1")
