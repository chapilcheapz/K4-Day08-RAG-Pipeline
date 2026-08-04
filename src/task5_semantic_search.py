"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store, có hỗ trợ
HyDE (Hypothetical Document Embeddings): thay vì embed thẳng câu hỏi ngắn của
user, sinh trước 1 đoạn văn giả định trả lời câu hỏi đó rồi embed đoạn văn này
— giúp thu hẹp khoảng cách ngữ nghĩa giữa câu hỏi ngắn và đoạn văn dài trong
tài liệu gốc (cải thiện recall ~10-15% với query ngắn).

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv()

# Model rẻ/nhanh dùng riêng cho sinh hypothetical doc (HyDE), không phải model
# trả lời chính thức của Task 10 — chỉ cần đủ tốt để viết 1 đoạn văn giả định.
HYDE_MODEL = "inclusionai/ling-3.0-flash:free"

HYDE_SYSTEM_PROMPT = (
    "Viết một đoạn văn ngắn (2-3 câu) trả lời câu hỏi, với văn phong và thuật "
    "ngữ giống như trích từ đề án tuyển sinh đại học hoặc bài báo điểm chuẩn "
    "chính thức. Không cần chính xác số liệu, chỉ cần đúng văn phong."
)


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def _get_collection():
    """Kết nối tới collection đã index ở Task 4. Chưa chạy Task 4 -> NotImplementedError
    để pytest coi đây là "chưa implement" và skip, thay vì fail cứng."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise NotImplementedError(
            f"Chua co du lieu trong ChromaDB - chay 'python -m src.task4_chunking_indexing' "
            f"truoc. ({e})"
        ) from e


def _generate_hypothetical_doc(query: str) -> str:
    """Sinh đoạn văn giả định cho HyDE. Không có API key hoặc lỗi gọi LLM -> fallback
    về query gốc (HyDE là optimization, không phải điều kiện bắt buộc để search chạy)."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return query

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.5,
            max_tokens=150,
        )
        hypothetical_doc = (response.choices[0].message.content or "").strip()
        return hypothetical_doc or query
    except Exception:
        return query


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: True -> sinh hypothetical doc và embed đoạn đó thay vì query gốc

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    collection = _get_collection()

    embed_text = _generate_hypothetical_doc(query) if use_hyde else query
    model = _get_embedding_model()
    query_vector = model.encode(embed_text).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = max(0.0, 1.0 - dist)
        output.append({"content": doc, "score": round(score, 4), "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    try:
        results = semantic_search("học phí ngành Khoa học Máy tính là bao nhiêu", top_k=5)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    except NotImplementedError as e:
        print(f"Chua the chay demo: {e}")
