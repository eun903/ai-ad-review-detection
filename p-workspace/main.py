from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import mysql.connector
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI(title="Ad Review Analyzer API")

logging.basicConfig(level=logging.INFO)

# 1. 파인튜닝된 모델 로드
logging.info("🚀 Loading fine-tuned MiniLM model...")
model = SentenceTransformer("./finetuned_minilm_model")

# 2. DB 설정
DB_CONFIG = {
    "host": "localhost",
    "database": "review_similarity",
    "user": "root",
    "password": "onlyroot",
    "auth_plugin": "mysql_native_password",
    "charset": "utf8mb4"
}

# 3. 주요 키워드 정의
AD_KEYWORDS = ['무료', '무료제공', '리뷰어', '체험단', '무상', '협찬', '이벤트', '제품제공']
NON_AD_KEYWORDS = ['내돈내산', '직접 구매', '직접 구입', '과장광고', '그나마', '별로', '너무 비싸']
NEGATIVE_PHRASES = ['별로', '실망', '비쌈', '후회', '돈 아까움', '비추', '안 좋음', '안 먹음']


# DB 연결 함수
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


# 요청 모델
class ReviewRequest(BaseModel):
    review: Optional[str] = None
    userReview: Optional[str] = None
    category: Optional[str] = None


# 응답 모델
class ReviewResponse(BaseModel):
    입력_리뷰: str
    가장_유사한_광고_리뷰: str
    유사도_점수: float
    label: int
    후보_리뷰들: list
    판단: str
    광고_키워드: list
    비광고_키워드: list


# # ✅ 키워드 매칭 함수
# def match_keywords(text: str, keyword_list: list) -> list:
#     return [kw for kw in keyword_list if re.search(r'\b' + re.escape(kw) + r'\b', text)]

def match_keywords(text: str, keyword_list: list) -> list:
    results = []
    
    for kw in keyword_list:
        pattern = re.escape(kw)
        
        if re.search(pattern, text, re.I):
            results.append(kw)
            
    return results

# ✅ 유효성 검증
def is_valid_review(text: str) -> bool:
    if text is None:
        return False
    text = text.strip()
    if len(text) < 5 or text.isdigit():
        return False
    if all(ch in "!@#$%^&*()_+=-[]{};:'\",.<>?/|" for ch in text):
        return False
    return True


# ✅ 안전한 코사인 유사도 계산
def safe_cosine_similarity(vec1, vec2):
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


# ✅ 핵심 분석 함수
def analyze_review(user_review: str, category: Optional[str] = None, top_n=3, threshold=0.7, min_similarity=0.2):
    matched_ad_keywords = match_keywords(user_review, AD_KEYWORDS)
    matched_non_ad_keywords = match_keywords(user_review, NON_AD_KEYWORDS)
    matched_negative_phrases = match_keywords(user_review, NEGATIVE_PHRASES)

    # 리뷰 검증
    if not is_valid_review(user_review):
        return {
            "입력_리뷰": user_review or "",
            "가장_유사한_광고_리뷰": "",
            "유사도_점수": 0.0,
            "label": -1,
            "후보_리뷰들": [],
            "판단": "분석 불가 (리뷰 내용 부족)",
            "광고_키워드": matched_ad_keywords,
            "비광고_키워드": matched_non_ad_keywords + matched_negative_phrases
        }

    # 1️⃣ 사용자 리뷰 벡터화
    user_vec = model.encode(user_review)

    # 2️⃣ DB 조회
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    if category:
        category = category.strip()  # ✅ 공백 제거
        logging.info(f"[DB] category filter applied: {category}")
        cur.execute("""
            SELECT cleaned_review, label, review_vector, category
            FROM reviews
            WHERE review_vector IS NOT NULL
            AND review_vector != ''
            AND TRIM(category) = %s
            AND label = 1  -- ✅ 광고 리뷰만 비교
        """, (category,))
    else:
        logging.info("[DB] no category provided — loading all ad reviews")
        cur.execute("""
            SELECT cleaned_review, label, review_vector, category
            FROM reviews
            WHERE review_vector IS NOT NULL
            AND review_vector != ''
            AND label = 1
        """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    logging.info(f"[DB] {len(rows)}개 후보 로드됨 (카테고리={category})")
    if rows:
        categories = set([r["category"] for r in rows])
        logging.info(f"[DB] 후보 카테고리 샘플: {list(categories)[:5]}")

    if not rows:
        return {
            "입력_리뷰": user_review,
            "가장_유사한_광고_리뷰": "",
            "유사도_점수": 0.0,
            "label": -1,
            "후보_리뷰들": [],
            "판단": "데이터 부족 (해당 카테고리 없음)",
            "광고_키워드": matched_ad_keywords,
            "비광고_키워드": matched_non_ad_keywords + matched_negative_phrases
        }

    # 3️⃣ 유사도 계산
    candidates = []
    best_score = -1.0
    best_review = ""
    best_label = -1

    for row in rows:
        try:
            review_vec = np.array(json.loads(row["review_vector"]))
            score = safe_cosine_similarity(user_vec, review_vec)
            candidates.append({"review": row["cleaned_review"], "score": float(score), "label": row["label"]})
            if score > best_score:
                best_score = score
                best_review = row["cleaned_review"]
                best_label = row["label"]
        except Exception as e:
            logging.warning(f"⚠️ 벡터 파싱 에러: {e}")
            continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = candidates[:top_n]

    # 4️⃣ 키워드 기반 보정
    keyword_adjustment = (
        0.05 * len(matched_ad_keywords)
        - 0.1 * len(matched_non_ad_keywords)
        - 0.1 * len(matched_negative_phrases)
    )
    final_score = max(0, min(1, best_score + keyword_adjustment))

    # 5️⃣ 판단 로직
    if final_score < min_similarity:
        decision_text = "광고성이 아닐 가능성이 높음"
        best_label = 0
    elif final_score >= threshold:
        decision_text = "광고성 리뷰일 가능성 높음"
        best_label = 1
    else:
        decision_text = "일반 리뷰일 가능성 높음"
        best_label = 0

    # 6️⃣ 결과 반환
    return {
        "입력_리뷰": user_review,
        "가장_유사한_광고_리뷰": best_review,
        "유사도_점수": round(float(final_score) * 100, 2),
        "label": int(best_label),
        "후보_리뷰들": [{"review": c["review"], "score": round(c["score"]*100, 2)} for c in top_candidates],
        "판단": decision_text,
        "광고_키워드": matched_ad_keywords,
        "비광고_키워드": matched_non_ad_keywords + matched_negative_phrases
    }


# FastAPI 기본 라우트
@app.get("/")
def root():
    return {"message": "FastAPI 서버 정상 작동 중 🚀"}


# 분석 요청
@app.post("/analyze", response_model=ReviewResponse)
def analyze(data: ReviewRequest):
    text = (data.review or data.userReview or "").strip()
    category = data.category
    logging.info(f"📦 받은 category 값: {category}")
    return analyze_review(text, category)
