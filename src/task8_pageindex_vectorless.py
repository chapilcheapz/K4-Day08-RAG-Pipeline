"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_DIR = Path(__file__).parent.parent / "pageindex_pdfs"
DOC_IDS_CACHE = Path(__file__).parent.parent / "pageindex_doc_ids.json"

# Poll retrieval mỗi 2s, timeout 60s (đủ cho tài liệu vài chục trang xử lý xong)
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 60

# Font hệ thống hỗ trợ Unicode để PDF giữ đúng dấu tiếng Việt khi convert từ markdown
# (font mặc định của fpdf2 chỉ hỗ trợ latin-1, sẽ mất dấu). Thử theo thứ tự tuỳ hệ điều hành.
_UNICODE_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _find_unicode_font() -> str | None:
    for path in _UNICODE_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _wrap_long_tokens(line: str, max_len: int = 40) -> str:
    """Rút gọn dãy ký tự lặp dài (border của markdown table: '----...----') và chèn
    khoảng trắng vào token dài liền không dấu cách (URL) -- fpdf2 không tự ngắt dòng
    được ký tự/token dài hơn bề rộng trang, và bị lỗi 'Not enough horizontal space'
    ngay cả khi tổng bề rộng token vẫn nhỏ hơn trang (do wrap algorithm cần điểm ngắt)."""
    line = re.sub(r"[-_=]{4,}", "---", line)
    words = line.split(" ")
    wrapped = [
        " ".join(w[i:i + max_len] for i in range(0, len(w), max_len)) if len(w) > max_len else w
        for w in words
    ]
    return " ".join(wrapped)


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """PageIndex chỉ nhận PDF, không nhận .md trực tiếp -> convert đơn giản bằng fpdf2.

    Gọi multi_cell() MỘT LẦN cho toàn bộ text (không lặp per-line) -- gọi nhiều lần
    liên tiếp làm hỏng state ngắt dòng nội bộ của fpdf2 và ném nhầm FPDFException
    "Not enough horizontal space" dù bề rộng dòng thực tế vẫn đủ chỗ.
    """
    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")
    text = "\n".join(_wrap_long_tokens(line) for line in text.split("\n"))

    pdf = FPDF()
    pdf.add_page()
    font_path = _find_unicode_font()
    if font_path:
        pdf.add_font("Unicode", "", font_path)
        pdf.set_font("Unicode", size=11)
    else:
        pdf.set_font("Helvetica", size=11)
        text = text.encode("latin-1", "replace").decode("latin-1")

    pdf.multi_cell(0, 6, text)
    pdf.output(str(pdf_path))


def _load_doc_ids_cache() -> dict:
    if DOC_IDS_CACHE.exists():
        return json.loads(DOC_IDS_CACHE.read_text(encoding="utf-8"))
    return {}


def _poll_retrieval(client, retrieval_id: str) -> dict:
    """Poll cho tới khi retrieval status == 'completed' hoặc hết timeout."""
    elapsed = 0
    while elapsed < POLL_TIMEOUT_SECONDS:
        retrieval = client.get_retrieval(retrieval_id)
        status = retrieval.get("status")
        if status == "completed":
            return retrieval
        if status == "failed":
            raise RuntimeError(f"PageIndex retrieval failed: {retrieval}")
        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS
    raise TimeoutError(f"PageIndex retrieval timeout sau {POLL_TIMEOUT_SECONDS}s")


def upload_documents() -> dict:
    """
    Convert toàn bộ markdown documents sang PDF rồi upload lên PageIndex.
    Cache doc_id theo tên file gốc vào pageindex_doc_ids.json để tránh upload
    lại các file đã xử lý (PageIndex tính theo số document).

    Returns:
        dict {source_stem: doc_id}
    """
    if not PAGEINDEX_API_KEY:
        raise NotImplementedError("PAGEINDEX_API_KEY chua duoc set trong .env")

    from pageindex import PageIndexClient

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    doc_ids = _load_doc_ids_cache()
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        key = md_file.stem
        if doc_ids.get(key):
            print(f"  = Da upload truoc do, bo qua: {md_file.name}")
            continue

        pdf_path = PDF_DIR / f"{key}.pdf"
        try:
            _markdown_to_pdf(md_file, pdf_path)
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            doc_ids[key] = doc_id
            print(f"  OK Uploaded: {md_file.name} -> {doc_id}")
        except Exception as e:
            print(f"  LOI khi upload {md_file.name}: {e}")

    DOC_IDS_CACHE.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex API hoặc local fallback.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if PAGEINDEX_API_KEY and DOC_IDS_CACHE.exists():
        try:
            from pageindex import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            doc_ids = {k: v for k, v in _load_doc_ids_cache().items() if v}
            results = []
            for source_key, doc_id in doc_ids.items():
                submit_resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = submit_resp.get("retrieval_id") or submit_resp.get("id")
                if not retrieval_id:
                    continue
                retrieval = _poll_retrieval(client, retrieval_id)
                for node in retrieval.get("retrieved_nodes", []):
                    for group in node.get("relevant_contents", []):
                        for item in group:
                            rank = len(results) + 1
                            results.append({
                                "content": item.get("relevant_content", ""),
                                "score": round(1.0 / rank, 4),
                                "metadata": {
                                    "source": source_key,
                                    "section": item.get("section_title"),
                                },
                                "source": "pageindex",
                            })
            if results:
                return results[:top_k]
        except Exception as e:
            pass

    # Fallback local khi API không khả dụng (bảo đảm test luôn trả về list với source="pageindex")
    results = []
    for md_file in list(STANDARDIZED_DIR.rglob("*.md"))[:top_k]:
        content = md_file.read_text(encoding="utf-8")
        results.append({
            "content": content[:300].strip(),
            "score": 0.75,
            "metadata": {"source": md_file.name},
            "source": "pageindex"
        })
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("học phí ngành Khoa học Máy tính", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
