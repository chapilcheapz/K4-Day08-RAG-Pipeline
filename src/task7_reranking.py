"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import math
from typing import Optional, Union


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Cross-Encoder model.
    Fallback tự động về sắp xếp theo score nếu không tải được model local/API key.
    """
    if not candidates:
        return []

    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, c["content"]] for c in candidates]
        scores = model.predict(pairs)
        
        reranked = []
        for cand, score in zip(candidates, scores):
            item = cand.copy()
            item["score"] = float(round(score, 4))
            reranked.append(item)
        
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
    except Exception:
        # Fallback về sắp xếp theo điểm hiện có
        sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_cands[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            cand_emb = candidates[idx].get("embedding")
            if not cand_emb:
                relevance = candidates[idx].get("score", 0.0)
            else:
                relevance = _cosine_similarity(query_embedding, cand_emb)

            max_sim_to_selected = 0.0
            if selected_indices and cand_emb:
                for sel_idx in selected_indices:
                    sel_emb = candidates[sel_idx].get("embedding")
                    if sel_emb:
                        sim = _cosine_similarity(cand_emb, sel_emb)
                        max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker (ví dụ: Dense + Sparse).
    Công thức: RRF(d) = Σ 1 / (k + rank_r(d))
    """
    if not ranked_lists:
        return []

    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: Union[list[dict], list[list[dict]]],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "rrf":
        if candidates and isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k=top_k)
        else:
            return rerank_rrf([candidates], top_k=top_k)
    elif method == "mmr":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = model.encode(query).tolist()
        cands_list = candidates[0] if candidates and isinstance(candidates[0], list) else candidates
        for c in cands_list:
            if "embedding" not in c:
                c["embedding"] = model.encode(c["content"]).tolist()
        return rerank_mmr(query_emb, cands_list, top_k=top_k)
    elif method == "cross_encoder":
        cands_list = candidates[0] if candidates and isinstance(candidates[0], list) else candidates
        return rerank_cross_encoder(query, cands_list, top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test thử nghiệm RRF với tuyển sinh
    list1 = [
        {"content": "Chính sách ưu tiên xét tuyển thẳng Đại học Bách khoa Hà Nội 2026", "score": 0.85, "metadata": {"source": "hust.md"}},
        {"content": "Điểm chuẩn trúng tuyển các ngành Khoa học Máy tính 2025", "score": 0.75, "metadata": {"source": "hust.md"}},
    ]
    list2 = [
        {"content": "Điểm chuẩn trúng tuyển các ngành Khoa học Máy tính 2025", "score": 9.5, "metadata": {"source": "hust.md"}},
        {"content": "Chính sách ưu tiên xét tuyển thẳng Đại học Bách khoa Hà Nội 2026", "score": 8.1, "metadata": {"source": "hust.md"}},
    ]
    results = rerank("xét tuyển bách khoa", [list1, list2], top_k=2, method="rrf")
    print("✓ Output RRF Reranking:")
    for r in results:
        print(f"  [{r['score']:.5f}] {r['content']}")

