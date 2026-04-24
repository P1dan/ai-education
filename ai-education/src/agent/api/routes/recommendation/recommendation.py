import os
import json
import random
import sqlite3
import requests
import warnings
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

# 抑制警告
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

DB_PATH = os.path.join(os.path.dirname(__file__), "Recommendation.db")

os.environ.setdefault("OPENAI_API_KEY", "sk-58c077b8242248dd8af6bfbe85431ba0")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

class RecommendationRequest(BaseModel):
    user_id: str

class ValidateUserRequest(BaseModel):
    user_id: str

class ValidateUserResponse(BaseModel):
    valid: bool
    message: str = ""

class RecommendationItem(BaseModel):
    name: str
    url: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

@dataclass
class UserProfile:
    major: str = ""
    enrollment_year: str = ""
    interest_long_profile: str = ""
    interest_short_profile: str = ""

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def check_user_exists_in_db(userid: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        try:
            cursor.execute(
                "SELECT 1 FROM student_model WHERE field1 = ? AND field1 != 'student_id' LIMIT 1",
                (userid,)
            )
            if cursor.fetchone():
                return True
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute(
                "SELECT 1 FROM interaction_records WHERE field2 = ? AND field2 != 'student_id' LIMIT 1",
                (userid,)
            )
            if cursor.fetchone():
                return True
        except sqlite3.OperationalError:
            pass
        return False
    finally:
        conn.close()

def find_history_interactions_from_db(userid: str) -> List[int]:
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT field3
        FROM interaction_records
        WHERE field2 = ? AND field2 != 'student_id'
    """
    cursor.execute(query, (userid,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        try:
            class_id = int(row[0])
            history.append(class_id)
        except Exception:
            continue
    return history

def find_user_profiles_from_db(userid: str) -> UserProfile:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT field5, field6, field9, field10
               FROM student_model
               WHERE field1 = ? AND field1 != 'student_id'""",
            (userid,)
        )
        row = cursor.fetchone()
        if row:
            profile = UserProfile(
                major=row[0] or "",
                enrollment_year=str(row[1]) if row[1] else "",
                interest_short_profile=row[2] or "",
                interest_long_profile=row[3] or ""
            )
        else:
            profile = UserProfile()
    except sqlite3.OperationalError:
        profile = UserProfile()
    finally:
        conn.close()
    return profile

def find_items_info_from_db() -> Tuple[
    List[int], Dict[int, str], Dict[int, Set[str]], Dict[int, Set[str]],
    Dict[int, str], Dict[int, str], Dict[int, np.ndarray]
]:
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT field1, field2, field3, field4, field5, field6, field7
        FROM class_index
        WHERE field1 != 'class_id'
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    item_pool = []
    item_names = {}
    item_keywords_pos = {}
    item_keywords_neg = {}
    item_content = {}
    item_url = {}
    item_bert_vectors: Dict[int, np.ndarray] = {}

    for row in rows:
        try:
            item_id = int(row[0])
            item_pool.append(item_id)
            item_names[item_id] = row[1] or f"课程{item_id}"
            item_content[item_id] = row[2] or ""
            item_url[item_id] = row[5] or "https://www.example.com"

            pos_kw = row[3]
            neg_kw = row[4]
            if pos_kw:
                try:
                    item_keywords_pos[item_id] = set(json.loads(pos_kw))
                except Exception:
                    item_keywords_pos[item_id] = set(pos_kw.split(',')) if isinstance(pos_kw, str) else set()
            else:
                item_keywords_pos[item_id] = set()

            if neg_kw:
                try:
                    item_keywords_neg[item_id] = set(json.loads(neg_kw))
                except Exception:
                    item_keywords_neg[item_id] = set(neg_kw.split(',')) if isinstance(neg_kw, str) else set()
            else:
                item_keywords_neg[item_id] = set()

            bert_vec_raw = row[6]
            if bert_vec_raw:
                try:
                    vec_list = json.loads(bert_vec_raw)
                    item_bert_vectors[item_id] = np.array(vec_list, dtype=np.float32)
                except Exception:
                    item_bert_vectors[item_id] = None
            else:
                item_bert_vectors[item_id] = None
        except Exception:
            continue

    return item_pool, item_names, item_keywords_pos, item_keywords_neg, item_content, item_url, item_bert_vectors

def load_interaction_bert_vectors(userid: str) -> Dict[int, np.ndarray]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT field3, field8
           FROM interaction_records
           WHERE field2 = ? AND field2 != 'student_id'""",
        (userid,)
    )
    rows = cursor.fetchall()
    conn.close()

    history_vectors: Dict[int, np.ndarray] = {}
    for row in rows:
        try:
            class_id = int(row[0])
            bert_vec_raw = row[1]
            if bert_vec_raw:
                vec_list = json.loads(bert_vec_raw)
                history_vectors[class_id] = np.array(vec_list, dtype=np.float32)
        except Exception:
            continue
    return history_vectors

def llm_fallback_recommendations(item_pool: List[int], item_names: Dict[int, str],
                                 item_url: Dict[int, str], k: int = 5) -> List[Dict[str, str]]:
    valid_items = [
        i for i in item_pool
        if item_names.get(i, "").strip() and item_url.get(i, "").strip()
    ]
    if not valid_items:
        return []
    selected = random.sample(valid_items, min(k, len(valid_items)))
    result = [{"name": item_names[i], "url": item_url[i]} for i in selected]
    return result

def f_mat(user_history_labels: List[Tuple[int, int]], item: int,
          item_keywords_pos: Dict[int, Set[str]], item_keywords_neg: Dict[int, Set[str]]) -> float:
    pos_score = 0
    neg_score = 0
    for hist_item, label in user_history_labels:
        if label == 1:
            pos_overlap = len(item_keywords_pos.get(item, set()) & item_keywords_pos.get(hist_item, set()))
            neg_overlap = len(item_keywords_neg.get(item, set()) & item_keywords_neg.get(hist_item, set()))
            pos_score += pos_overlap - neg_overlap
        else:
            pos_overlap = len(item_keywords_pos.get(item, set()) & item_keywords_neg.get(hist_item, set()))
            neg_overlap = len(item_keywords_neg.get(item, set()) & item_keywords_pos.get(hist_item, set()))
            neg_score += pos_overlap - neg_overlap
    return pos_score - neg_score

def f_sim_precomputed(user_history_labels: List[Tuple[int, int]], item: int,
                      item_bert_vectors: Dict[int, np.ndarray]) -> float:
    item_vec = item_bert_vectors.get(item)
    if item_vec is None:
        return 0.0

    sim_scores = []
    for hist_item, label in user_history_labels:
        hist_vec = item_bert_vectors.get(hist_item)
        if hist_vec is None:
            continue
        sim = cosine_similarity(item_vec.reshape(1, -1), hist_vec.reshape(1, -1))[0][0]
        sim_scores.append(float(sim) * (1 if label == 1 else -1))

    return sum(sim_scores) if sim_scores else 0.0

def generate_llm_recommendations(user_history: List[int], item_names: Dict[int, str],
                                 user_profile: UserProfile, user_id: str, k: int = 5) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://xiaoai.plus/v1")

    history_names = [item_names.get(item_id, f"课程{item_id}") for item_id in user_history]

    prompt = f"""你是一个专业的课程推荐助手。请为以下用户推荐{k}门合适的课程。

用户信息：
- 专业：{user_profile.major or '未指定'}
- 兴趣：{user_profile.interest_long_profile or '未指定'}
- 已修课程：{', '.join(history_names) if history_names else '无'}

请根据用户的专业背景和兴趣，生成个性化的课程推荐理由。"""

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(
        api_key = api_key,
        base_url = base_url,
        model = "deepseek-chat",
        temperature = 0.7
    )


    try:
        ai_response = llm.invoke([HumanMessage(content=prompt)])
        result = ai_response.content  # 这已经是字符串内容了
        return result
    except Exception as e:
        print(f"LLM error: {e}")
        # 返回降级推荐的课程列表（建议返回结构化数据）
        return [f"课程{i}" for i in range(1, k+1)]

def match_generated_content_to_items(generated_content: str, candidate_items: List[int],
                                     item_content: Dict[int, str],
                                     item_bert_vectors: Dict[int, np.ndarray]) -> List[Tuple[int, float]]:
    try:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'sentence-transformers', 'all-mpnet-base-v2')
        encoder = SentenceTransformer(model_path)
        generated_embedding = encoder.encode([generated_content])

        valid_items = []
        valid_embeddings = []
        for item_id in candidate_items:
            vec = item_bert_vectors.get(item_id)
            if vec is not None:
                valid_items.append(item_id)
                valid_embeddings.append(vec)

        if not valid_embeddings:
            return [(item, random.random()) for item in candidate_items]

        item_embeddings = np.vstack(valid_embeddings)
        similarities = cosine_similarity(generated_embedding, item_embeddings)[0]

        score_dict = dict(zip(valid_items, similarities))

        result = []
        for item in candidate_items:
            if item in score_dict:
                result.append((item, float(score_dict[item])))
            else:
                result.append((item, random.random()))
        return result
    except Exception as e:
        print(f"Content matching error: {e}")
        return [(item, random.random()) for item in candidate_items]

def recommend_top_k_precomputed(
        user_history: List[int], item_pool: List[int],
        item_names: Dict[int, str], item_keywords_pos: Dict[int, Set[str]],
        item_keywords_neg: Dict[int, Set[str]], item_content: Dict[int, str],
        item_url: Dict[int, str], item_bert_vectors: Dict[int, np.ndarray],
        user_profile: UserProfile, user_id: str, k: int = 5,
        alpha: float = 0.2, beta: float = 0, gamma: float = 0.8
) -> Tuple[List[str], List[str]]:

    interacted_items = set(user_history)
    candidate_items = [item for item in item_pool if item not in interacted_items]

    if not candidate_items:
        top_items = item_pool[:k]
        return ([item_names.get(i, "未知课程") for i in top_items],
                [item_url.get(i, "") for i in top_items])

    user_history_labels = [(item, 1) for item in user_history]
    num_neg = min(len(user_history), len(candidate_items))
    if num_neg > 0:
        neg_items = random.sample(candidate_items, num_neg)
        user_history_labels.extend([(item, 0) for item in neg_items])

    mat_scores = [f_mat(user_history_labels, item, item_keywords_pos, item_keywords_neg)
                  for item in candidate_items]
    sim_scores = [f_sim_precomputed(user_history_labels, item, item_bert_vectors)
                  for item in candidate_items]

    generated_content = generate_llm_recommendations(user_history, item_names, user_profile, user_id, k)
    llm_similarities = match_generated_content_to_items(generated_content, candidate_items,
                                                        item_content, item_bert_vectors)
    llm_score_dict = dict(llm_similarities)
    llm_scores = [llm_score_dict.get(item, 0.0) for item in candidate_items]

    def normalize(scores):
        if not scores or max(scores) == min(scores):
            return [0.0] * len(scores)
        return [(s - min(scores)) / (max(scores) - min(scores)) for s in scores]

    mat_norm = normalize(mat_scores)
    sim_norm = normalize(sim_scores)
    llm_norm = normalize(llm_scores)

    total_scores = [
        alpha * m + beta * s + gamma * l
        for m, s, l in zip(mat_norm, sim_norm, llm_norm)
    ]
    scored_items = sorted(zip(candidate_items, total_scores), key=lambda x: x[1], reverse=True)
    top_k = scored_items[:k]

    top_k_names = [item_names.get(item, "未知课程") for item, _ in top_k]
    top_k_urls = [item_url.get(item, "") for item, _ in top_k]

    return top_k_names, top_k_urls

def recommender(userid: str, topk: int = 5) -> List[Dict[str, str]]:
    history = find_history_interactions_from_db(userid)
    profile = find_user_profiles_from_db(userid)
    item_pool, item_names, item_keywords_pos, item_keywords_neg, \
        item_content, item_url, item_bert_vectors = find_items_info_from_db()

    if not item_pool:
        return []

    if history:
        top_k_names, top_k_urls = recommend_top_k_precomputed(
            history, item_pool, item_names, item_keywords_pos, item_keywords_neg,
            item_content, item_url, item_bert_vectors, profile, userid, k=topk
        )
    else:
        top_items = item_pool[:topk]
        top_k_names = [item_names.get(item, "未知课程") for item in top_items]
        top_k_urls = [item_url.get(item, "") for item in top_items]

    recommendations = [
        {"name": name, "url": url}
        for name, url in zip(top_k_names, top_k_urls)
        if name and name.strip() and url and url.strip()
    ]

    if len(recommendations) < topk:
        existing_names = {r["name"] for r in recommendations}
        fallback = llm_fallback_recommendations(item_pool, item_names, item_url, k=topk * 2)
        for fb in fallback:
            if fb["name"] not in existing_names:
                recommendations.append(fb)
                existing_names.add(fb["name"])
            if len(recommendations) >= topk:
                break

    return recommendations




router = APIRouter()

@router.get("/", tags=["系统"])
async def root():
    return {
        "message": "个性化课程推荐服务",
        "status": "healthy",
        "version": "2.0.0",
        "docs": "/docs"
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="个性化课程推荐",
        version="2.0.0"
    )

@router.post("/validate_user", response_model=ValidateUserResponse)
async def validate_user(request: ValidateUserRequest):
    user_id = request.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="学号不能为空")
    exists = check_user_exists_in_db(user_id)
    if exists:
        return ValidateUserResponse(valid=True, message="学号验证通过")
    raise HTTPException(status_code=404, detail=f"学号 {user_id} 不存在")

@router.post("/recommendations", response_model=List[RecommendationItem])
async def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = recommender(request.user_id, topk=5)
        return recommendations
    except Exception as e:
        print(f"Recommendation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")

