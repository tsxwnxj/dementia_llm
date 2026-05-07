from fastapi import HTTPException, Header
from firebase_admin import auth
import firebase_admin

async def verify_firebase_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")

    token = authorization.split("Bearer ")[1]

    # Firebase 미초기화 시 개발 모드로 통과 (Mock 테스트용)
    if not firebase_admin._apps:
        return {"uid": "dev_user", "dev_mode": True}

    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
