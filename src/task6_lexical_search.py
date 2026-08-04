"""
Task 6 — Lexical Search Module (BM25 & TF-IDF).

Mặc định sử dụng BM25. Cung cấp thêm TF-IDF để so sánh độ chính xác từ khóa.
Cơ chế hoạt động:
    - BM25: Chấm điểm dựa trên Term Frequency (TF), Inverse Document Frequency (IDF)
      và bình thường hóa độ dài tài liệu (Document Length Normalization).
      k1 = 1.5 (độ bão hòa từ khóa), b = 0.75 (mức độ phạt tài liệu dài).
    - TF-IDF: Chấm điểm dựa trên biểu diễn vector tần suất và tính tương đồng Cosine.
"""

import os
from pathlib import Path

# Thư mục chứa dữ liệu đã chuẩn hóa
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Corpus lưu trữ dưới dạng danh sách các dict: {'content': str, 'metadata': dict}
CORPUS: list[dict] = []

# Lưu trữ các chỉ mục sau khi xây dựng xong
BM25_INDEX = None
TFIDF_VECTORIZER = None
TFIDF_MATRIX = None


def load_corpus() -> list[dict]:
    """
    Tải corpus làm đầu vào cho BM25 & TF-IDF.
    Ưu tiên: Tải trực tiếp 100% các chunks đã được lập chỉ mục trong ChromaDB (Task 4)
    để đảm bảo đồng bộ tuyệt đối với Semantic Search (Task 5).
    Nếu ChromaDB trống hoặc lỗi, tự động fallback đọc file và chunking theo cấu hình chuẩn.
    """
    corpus = []
    
    # 1. Thử tải trực tiếp từ ChromaDB (ưu tiên số 1)
    try:
        import chromadb
        chroma_dir = Path(__file__).parent.parent / "chroma_db"
        if chroma_dir.exists():
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_collection(name="university_admissions_docs")
            results = collection.get(include=["documents", "metadatas"])
            if results and "documents" in results and results["documents"]:
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    corpus.append({
                        "content": doc,
                        "metadata": meta
                    })
                print(f"✓ Đã đồng bộ thành công {len(corpus)} chunks trực tiếp từ ChromaDB.")
                return corpus
    except Exception as e:
        print(f"ℹ️ Không thể lấy chunks từ ChromaDB ({e}), đang chuyển sang đọc file thô...")

    # 2. Fallback đọc file & chunking (nếu DB chưa sẵn sàng)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not STANDARDIZED_DIR.exists():
        print(f"⚠ Thư mục {STANDARDIZED_DIR} chưa tồn tại!")
        return corpus

    # Cấu hình RecursiveCharacterTextSplitter khớp tuyệt đối với cấu hình của Task 4
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Đọc đệ quy tất cả các file .md trong standardized/
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in md_file.as_posix() else "news"
            
            # Cắt tài liệu thành các chunks
            splits = splitter.split_text(content)
            for i, chunk_text in enumerate(splits):
                text_clean = chunk_text.strip()
                if text_clean:
                    corpus.append({
                        "content": text_clean,
                        "metadata": {
                            "source": md_file.name,
                            "type": doc_type,
                            "chunk_index": i
                        }
                    })
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {md_file}: {e}")

    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.
    k1 = 1.5 (term saturation), b = 0.75 (length normalization)
    """
    from rank_bm25 import BM25Okapi

    # Tokenize corpus (chuyển chữ thường và tách từ theo khoảng trắng)
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    return bm25


def build_tfidf_index(corpus: list[dict]):
    """
    Xây dựng TF-IDF index từ corpus sử dụng Scikit-learn.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    global TFIDF_VECTORIZER, TFIDF_MATRIX
    texts = [doc["content"] for doc in corpus]
    
    # Tạo vectorizer và fit corpus
    TFIDF_VECTORIZER = TfidfVectorizer(lowercase=True)
    TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(texts)


def init_indexes():
    """
    Lazy initialization: Chỉ tải corpus và dựng index khi thực hiện truy vấn lần đầu.
    """
    global CORPUS, BM25_INDEX, TFIDF_MATRIX
    if not CORPUS:
        CORPUS.extend(load_corpus())
        print(f"ℹ️ Đã tải {len(CORPUS)} chunks vào Corpus.")
        
    if CORPUS and BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(CORPUS)
        build_tfidf_index(CORPUS)
        print("ℹ️ Đã xây dựng thành công chỉ mục BM25 & TF-IDF.")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng thuật toán BM25.

    Args:
        query: Câu truy vấn từ khóa của người dùng
        top_k: Số lượng kết quả mong muốn tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
    """
    init_indexes()
    if not BM25_INDEX or not CORPUS:
        return []

    # Tokenize query
    tokenized_query = query.lower().split()
    scores = BM25_INDEX.get_scores(tokenized_query)

    # Lấy top_k kết quả có điểm cao nhất
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # Chỉ lấy các kết quả có điểm trùng khớp thực tế (> 0)
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng thuật toán TF-IDF và độ đo tương đồng Cosine.

    Args:
        query: Câu truy vấn từ khóa của người dùng
        top_k: Số lượng kết quả mong muốn tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # Cosine Similarity score [0, 1]
            'metadata': dict
        }
    """
    init_indexes()
    if TFIDF_MATRIX is None or TFIDF_VECTORIZER is None or not CORPUS:
        return []

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Vector hóa câu query
    query_vec = TFIDF_VECTORIZER.transform([query])

    # Tính độ tương đồng cosine giữa query và toàn bộ corpus
    similarity_scores = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()

    # Lấy top_k kết quả có độ tương đồng cao nhất
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarity_scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(similarity_scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test nhanh chức năng tìm kiếm
    print("--- Thử nghiệm tìm kiếm từ khóa tuyển sinh HUST ---")
    query_test = "IELTS xét tuyển thẳng Bách Khoa"
    
    print(f"\n[BM25] Truy vấn: '{query_test}'")
    bm25_res = lexical_search(query_test, top_k=3)
    for i, r in enumerate(bm25_res, 1):
        print(f"  {i}. [{r['score']:.3f}] Source: {r['metadata']['source']} | {r['content'][:150]}...")

    print(f"\n[TF-IDF] Truy vấn: '{query_test}'")
    tfidf_res = tfidf_search(query_test, top_k=3)
    for i, r in enumerate(tfidf_res, 1):
        print(f"  {i}. [Sim: {r['score']:.3f}] Source: {r['metadata']['source']} | {r['content'][:150]}...")
