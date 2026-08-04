"""
RAG Chatbot — Tuyển Sinh Đại Học 2026 (Chủ Đề 4)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cache RAG generation function để tối ưu hiệu năng UI
@st.cache_resource
def get_rag_generator():
    from src.task10_generation import generate_with_citation
    return generate_with_citation


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Trợ Lý Tư Vấn Tuyển Sinh Đại Học 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🎓 Tuyển Sinh Đại Học 2026")
    st.caption("Hệ thống trợ lý RAG tư vấn thông tin tuyển sinh, phương thức xét tuyển & điểm chuẩn các trường Đại học")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Phương thức xét tuyển thẳng của Đại học Bách Khoa Hà Nội 2026?",
        "Điểm chuẩn trúng tuyển các ngành Khoa học Máy tính 2025?",
        "Thông tin tuyển sinh đại học chính quy Trường ĐH KHTN?",
        "Học phí dự kiến và quy chế tuyển sinh Đại học Quốc gia Hà Nội?",
        "Điểm chuẩn trúng tuyển Đại học Kinh tế Quốc dân?",
    ]
    for idx, s in enumerate(suggestions):
        if st.button(s, use_container_width=True, key=f"sug_btn_{idx}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập RAG")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)

    st.divider()
    st.caption("**Kiến trúc hệ thống RAG v2 (Nhóm 6 người):**")
    st.caption("Dense (SentenceTransformers) + Sparse (BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 Trợ Lý Hỏi Đáp Tuyển Sinh Đại Học 2026")
st.caption("Hệ thống RAG Pipeline tìm kiếm & tổng hợp câu trả lời từ Đề án & Tin tức Tuyển sinh")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Tài liệu chưa rõ")
                    doc_type = meta.get("type", meta.get("doc_type", "legal/news"))
                    score = float(src.get("score", 0.0))
                    retrieval_src = src.get("source", "hybrid")
                    st.markdown(f"**[{i}] File Nguồn:** `{source_name}` | **Loại:** `{doc_type}` | **Rerank Score:** `{score:.4f}` | **Retrieval:** `{retrieval_src}`")
                    st.text(src.get("content", "")[:300] + "...")
                    st.divider()

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về đề án tuyển sinh, điểm chuẩn, ngành học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu cơ sở dữ liệu tuyển sinh và tổng hợp câu trả lời..."):
            try:
                generate_fn = get_rag_generator()
                response = generate_fn(query, top_k=top_k)
                answer = response.get("answer", "Tôi không thể xác minh thông tin này từ nguồn hiện có.")
                sources = response.get("sources", [])

            except NotImplementedError as e:
                answer = f"⚠️ **Một số mô-đun RAG chưa được khởi tạo:** {e}"
                sources = []
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Nguồn tham khảo chi tiết ({len(sources)} chunks)"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Tài liệu tuyển sinh")
                        doc_type = meta.get("type", meta.get("doc_type", "legal/news"))
                        score = float(src.get("score", 0.0))
                        retrieval_src = src.get("source", "hybrid")
                        st.markdown(f"**[{i}] File Nguồn:** `{source_name}` | **Loại:** `{doc_type}` | **Score:** `{score:.4f}` | **Kênh:** `{retrieval_src}`")
                        st.text(src.get("content", "")[:300] + "...")
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

