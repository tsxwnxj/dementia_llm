# HandFit - FastAPI 모델 서버

## 기술 스택
- FastAPI
- Docker
- Firebase Admin SDK

---

## 초기 환경 세팅

### 1. 레포 클론
\`\`\`bash
git clone [레포 주소]
cd server
\`\`\`

### 2. 필수 파일 추가
팀장에게 아래 파일을 받아서 루트 폴더에 넣어주세요.
- \`serviceAccountKey.json\`

### 3. Docker 설치
https://www.docker.com/products/docker-desktop/ 에서 설치

### 4. 서버 실행
\`\`\`bash
docker compose up --build
\`\`\`

### 5. 서버 확인
\`\`\`bash
curl http://localhost:8000/health
\`\`\`
정상 응답: \`{"status":"ok","service":"DementiaApp API"}\`

---

## API 엔드포인트
| Method | URL | 설명 |
|--------|-----|------|
| GET | /health | 서버 상태 확인 |
| POST | /api/v1/motion/analyze | 동작 분석 |

---

## 주의사항
- \`serviceAccountKey.json\` 은 절대 깃에 올리지 마세요
- 서버 실행 시 Docker Desktop이 켜져 있어야 합니다
