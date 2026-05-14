"""
사투리/노인 발음 정규화 파이프라인

Pipeline:
  raw STT text
    → 개인화 사전 (유저별 학습된 매핑)
    → 규칙 기반 패턴 (경상도/전라도/노인 발음)
    → 음소 유사도 기반 fuzzy 교정 (rapidfuzz)
    → 정규화된 텍스트

주요 처리:
- 경상도: 모음 변형 (ㅏ→ㅓ, ㅗ→ㅜ 경향), 어미 변형
- 전라도: 어미 변형 (-어/-아 → -이/-여)
- 노인 발음: 연음 탈락, 종성 혼동
- 외래어 한국어화: 일반적인 노인 발음 패턴
"""

import logging
import re
from typing import Optional

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


# ── 규칙 기반 패턴 사전 ────────────────────────────────────────────────────────
# 형식: (wrong_pattern, correct) — 정규식 패턴 가능

_RULE_PATTERNS: list[tuple[str, str]] = [
    # ── 경상도 방언 ────────────────────────────────────────────────────────────
    ("까자", "과자"),
    ("까제", "과제"),
    ("까일", "과일"),
    ("무따", "먹었다"),
    ("무겠다", "먹겠다"),
    ("묵었다", "먹었다"),
    ("묵자", "먹자"),
    ("가이소", "가세요"),
    ("오이소", "오세요"),
    ("하이소", "하세요"),
    ("마이", "많이"),
    ("머라", "뭐라고"),
    ("아이가", "아닌가"),
    ("아이다", "아니다"),
    ("와이라노", "왜 그러냐"),
    ("카더라", "그러더라"),
    ("카이", "그러니"),
    ("그카면", "그러면"),
    ("이카면", "이러면"),
    ("저카면", "저러면"),
    ("우야노", "어떡하냐"),
    ("우야꼬", "어떡하지"),

    # ── 전라도 방언 ────────────────────────────────────────────────────────────
    ("그랑께", "그러니까"),
    ("워메", "어머"),
    ("거시기", "그것"),
    ("요렇게", "이렇게"),
    ("조렇게", "저렇게"),
    ("글쎄잉", "글쎄요"),
    ("아니당께", "아니라니까요"),

    # ── 노인 발음 공통 패턴 ────────────────────────────────────────────────────
    ("으메", "어머"),
    ("에구", "에고"),
    ("아이고", "아이고"),      # 유지
    ("에이구", "에이고"),
    ("허참", "허참"),          # 유지
    ("어디가", "어디 가"),
    ("뭐해", "뭐 해"),
    ("이거이", "이것이"),
    ("저거이", "저것이"),
    ("그거이", "그것이"),
    ("집이서", "집에서"),
    ("밥이먹다", "밥을 먹다"),

    # ── 연음/받침 탈락 ─────────────────────────────────────────────────────────
    ("읽어", "읽어"),          # 유지
    ("닭이", "닭이"),          # 유지
    ("넘어", "넘어"),          # 유지

    # ── 흔한 오인식 패턴 (Whisper STT 오류) ────────────────────────────────────
    ("안녕하십니꺼", "안녕하세요"),
    ("감사하십니꺼", "감사합니다"),
    ("괜찮으십니꺼", "괜찮으세요"),
]

# 컴파일된 패턴 (초기화 시 1회)
_COMPILED_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(re.escape(wrong)), correct)
    for wrong, correct in _RULE_PATTERNS
]

# ── 어휘 교정 후보 사전 (fuzzy matching 대상) ─────────────────────────────────
# 일상적 한국어 명사 — STT 오인식 교정에 사용
_VOCAB_CANDIDATES: list[str] = [
    "과자", "과일", "가세요", "오세요", "많이", "먹었다", "먹자",
    "그러니까", "어머", "그러면", "어떡하냐", "아니다",
    "산책", "날씨", "꽃", "나무", "하늘", "강", "길", "공원",
    "점심", "저녁", "아침", "밥", "물", "약", "병원",
    "집", "가족", "아들", "딸", "손자", "손녀",
    "봄", "여름", "가을", "겨울",
    "기분", "건강", "피곤", "아파요", "좋아요",
]

# fuzzy 교정 최소 유사도 임계값 (0~100)
_FUZZY_THRESHOLD = 80
# fuzzy 교정 적용할 단어 최소 길이 (너무 짧은 단어는 오교정 위험)
_FUZZY_MIN_LEN = 3


def normalize(
    text: str,
    user_dict: Optional[dict[str, str]] = None,
) -> tuple[str, list[dict]]:
    """
    발음/사투리 정규화 메인 함수.

    Args:
        text: STT 원본 텍스트
        user_dict: 유저별 개인화 발음 매핑 {wrong: correct}

    Returns:
        (normalized_text, corrections)
        corrections: [{"original": "까자", "corrected": "과자", "method": "rule"}]
    """
    if not text or not text.strip():
        return text, []

    corrections: list[dict] = []
    current = text

    # 1단계: 개인화 사전 (최우선)
    if user_dict:
        current, c = _apply_dict(current, user_dict, method="personal")
        corrections.extend(c)

    # 2단계: 규칙 기반 패턴
    current, c = _apply_rules(current)
    corrections.extend(c)

    # 3단계: 단어 단위 fuzzy 교정
    current, c = _apply_fuzzy(current)
    corrections.extend(c)

    if corrections:
        logger.debug(
            "Pronunciation normalized: '%s' → '%s' (%d corrections)",
            text, current, len(corrections),
        )

    return current, corrections


def _apply_dict(
    text: str,
    dictionary: dict[str, str],
    method: str = "dict",
) -> tuple[str, list[dict]]:
    """사전 기반 단순 교체."""
    corrections: list[dict] = []
    result = text
    for wrong, correct in dictionary.items():
        if wrong in result:
            result = result.replace(wrong, correct)
            corrections.append({"original": wrong, "corrected": correct, "method": method})
    return result, corrections


def _apply_rules(text: str) -> tuple[str, list[dict]]:
    """컴파일된 정규식 규칙 적용."""
    corrections: list[dict] = []
    result = text
    for pattern, correct in _COMPILED_RULES:
        if pattern.search(result):
            new_result = pattern.sub(correct, result)
            if new_result != result:
                corrections.append({
                    "original": pattern.pattern,
                    "corrected": correct,
                    "method": "rule",
                })
                result = new_result
    return result, corrections


def _apply_fuzzy(text: str) -> tuple[str, list[dict]]:
    """
    단어 단위 fuzzy 교정.
    각 단어를 어휘 후보와 비교, 유사도가 높으면 교체.
    """
    corrections: list[dict] = []
    words = text.split()
    result_words: list[str] = []

    for word in words:
        # 너무 짧은 단어는 건너뜀 (오교정 방지)
        if len(word) < _FUZZY_MIN_LEN:
            result_words.append(word)
            continue

        # 이미 정확한 단어면 건너뜀
        if word in _VOCAB_CANDIDATES:
            result_words.append(word)
            continue

        match = process.extractOne(
            word,
            _VOCAB_CANDIDATES,
            scorer=fuzz.ratio,
            score_cutoff=_FUZZY_THRESHOLD,
        )

        if match:
            best_match, score, _ = match
            # 너무 다른 길이는 오교정 위험
            if abs(len(word) - len(best_match)) <= 2:
                corrections.append({
                    "original": word,
                    "corrected": best_match,
                    "method": f"fuzzy({score:.0f})",
                })
                result_words.append(best_match)
            else:
                result_words.append(word)
        else:
            result_words.append(word)

    return " ".join(result_words), corrections


def add_to_vocab(word: str) -> None:
    """동적으로 어휘 후보 추가 (개인화 확장용)."""
    if word and word not in _VOCAB_CANDIDATES:
        _VOCAB_CANDIDATES.append(word)
        logger.debug("Added to vocab candidates: %s", word)
