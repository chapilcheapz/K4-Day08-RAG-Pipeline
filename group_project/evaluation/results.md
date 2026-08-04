# 📊 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG RAGAS & A/B TESTING (GROUP PROJECT)

**Dự án**: Trợ Lý Tư Vấn Tuyển Sinh Đại Học 2026 (Chủ đề 4)  
**Nhóm**: 6 thành viên (Phân công đầy đủ Role 1 ➔ Role 6)  
**Bộ dữ liệu chuẩn (Golden Dataset)**: 20 câu Q&A Đề án tuyển sinh & Điểm chuẩn  

---

## 🏆 I. Bảng So Sánh A/B Testing (4 Chỉ Số RAGAS Industry Standard)

| Cấu Hình Pipeline (Config) | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Latency TB / câu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A: Hybrid Search (Dense + Sparse BM25 + RRF + PageIndex)** | **0.94** | **0.91** | **0.89** | **0.92** | **9.43s** |
| **Config B: Dense Search Only (Chỉ dùng Semantic Search)** | 0.81 | 0.76 | 0.72 | 0.78 | 0.05s |

---

## 👥 II. Phân Công Chi Tiết 6 Thành Viên Trong Nhóm

1. **Role 1 (Team Leader & Architect)**: Quản lý tổng thể dự án, chốt cấu hình Vector DB, đóng gói `app.py` & nộp bài.
2. **Role 2 (Data Scraper Dev)**: Thu thập dữ liệu Đề án/Bài báo tuyển sinh bằng `Crawl4AI`, convert PDF sang Markdown, đảm bảo metadata `source` & `doc_type`.
3. **Role 3 (Vector Search Dev)**: Lập chỉ mục ChromaDB (`all-MiniLM-L6-v2`), xây dựng pipeline nối chuỗi Dense + Sparse RRF Rerank (`src/task9_retrieval_pipeline.py`).
4. **Role 4 (Sparse Search Dev)**: Xây dựng module tìm kiếm từ khóa BM25, lập trình logic Fallback sang PageIndex (`pageindex_search`) khi score < 0.3.
5. **Role 5 (Frontend UI Dev)**: Xây dựng ứng dụng Streamlit Chatbot (`app.py`), thiết kế `st.expander` nguồn tham khảo & định dạng score `0.0000`.
6. **Role 6 (Benchmark QA Dev)**: Biên soạn Golden Dataset 20 câu Q&A tuyển sinh (`golden_dataset.json`), thực thi `eval_pipeline.py`, xuất báo cáo `results.md` và kiểm thử Pytest **35/35 PASSED**.

---

## 🎯 III. Kết Luận & Bài Học Kinh Nghiệm

1. **Hiệu quả của Hybrid Search + RRF**: Kết hợp Dense Search và BM25 mang lại độ phủ (Recall) tăng hơn 17% so với chỉ dùng Dense Search đơn thuần đối với các truy vấn chứa từ khóa viết tắt như "IELTS", "NEU", "HUST".
2. **Document Reordering**: Kỹ thuật đặt thông tin quan trọng ở đầu và cuối context giúp câu trả lời sinh ra từ LLM không bị trôi dữ liệu (Lost in the middle).
3. **Sẵn sàng Demo**: Hệ thống `app.py` đáp ứng mượt mà cả tìm kiếm ngữ nghĩa lẫn tra cứu từ khóa chính xác.
