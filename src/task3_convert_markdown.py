"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CLEANUP HELPERS
#
# MarkItDown/crawl4ai convert nguyên trang: menu điều hướng, footer, form liên
# hệ, link bài liên quan, quảng cáo — làm nhiễu nội dung khi chunking (Task 4).
# Loại các dòng rõ ràng là rác trước khi lưu, không cố "trích xuất nội dung
# chính" (rủi ro cắt nhầm nội dung thật vì mỗi trang web một cấu trúc khác nhau).
# =============================================================================

_JS_LINK_RE = re.compile(r"javascript:void", re.IGNORECASE)
_IMG_ONLY_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
_NAV_LINK_ONLY_RE = re.compile(r"^\s*[*\-]?\s*(!\[[^\]]*\]\([^)]*\))?\s*\[[^\]]*\]\([^)]*\)\s*$")
_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
_SPAM_DOMAIN_RE = re.compile(
    r"sv388|ga6789|tartanhockey|taixiu|sunwin|go88vip|okvip|\bbk8\b|tk88|ko66|"
    r"sv66|woot\.eu\.com|w88\.mov|789win|xocdia88|ev88\.com|68gamebai|nohu90|"
    r"kubet|mb66|98win|\bm88\b|b52\.com|hb88|99ok",
    re.IGNORECASE,
)
_NOISE_LINE_EXACT = {
    "×", "prev", "next", "tìm kiếm", "gửi", "hủy", "toggle navigation",
}
_NOISE_KEYWORDS = (
    "tổng truy cập", "follow us", "designed and developed by",
    "bản quyền thuộc", "copyright @", "copyright ",
)


def _clean_web_markdown(text: str) -> str:
    """Loại bỏ menu điều hướng, link JS, ảnh trần, domain spam khỏi markdown crawl."""
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _JS_LINK_RE.search(stripped):
            continue
        if _SPAM_DOMAIN_RE.search(stripped):
            continue
        if _IMG_ONLY_RE.match(stripped):
            continue
        if _NAV_LINK_ONLY_RE.match(stripped) and len(stripped) < 150:
            continue
        if stripped.lower() in _NOISE_LINE_EXACT:
            continue
        if any(kw in stripped.lower() for kw in _NOISE_KEYWORDS):
            continue
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _clean_pdf_markdown(text: str) -> str:
    """Loại bỏ số trang PDF đứng riêng 1 dòng (do pdfminer đọc theo layout trang)."""
    cleaned_lines = [
        line for line in text.split("\n")
        if not _PAGE_NUMBER_RE.match(line.strip())
    ]
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            if not result.text_content.strip():
                print(f"  SKIP: {filepath.name} co ve la PDF scan anh, khong extract duoc text")
                continue
            content = _clean_pdf_markdown(result.text_content)
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(content, encoding="utf-8")
            print(f"  OK Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            body = _clean_web_markdown(data.get("content_markdown", ""))
            content = header + body
            output_path.write_text(content, encoding="utf-8")
            print(f"  OK Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
