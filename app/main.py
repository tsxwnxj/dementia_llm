from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.routers import health, motion, walk_chat
import firebase_admin
from firebase_admin import credentials
import os

# Firebase Admin 초기화
if os.path.isfile("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="DementiaApp API", version="1.0.0")
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
