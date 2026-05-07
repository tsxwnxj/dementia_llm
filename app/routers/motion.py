from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.middleware.auth import verify_firebase_token
from app.models.schemas import MotionResult
from PIL import Image
import io

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_TYPES = {"image/jpeg", "image/png"}

@router.post("/motion/analyze", response_model=MotionResult)
@limiter.limit("30/minute")
async def analyze_motion(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(verify_firebase_token),
):
    # 파일 타입 검사
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="jpeg/png만 허용됩니다")

    contents = await file.read()

    # 파일 크기 검사
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="5MB 이하 이미지만 허용됩니다")

    # 이미지 유효성 검사
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="유효하지 않은 이미지입니다")

    # TODO: 실제 모델 교체 예정
    return MotionResult(
        is_correct=True,
        score=85,
        feedback="좋아요! 손가락을 조금 더 펴보세요.",
        exercise_type="finger_coordination",
    )
