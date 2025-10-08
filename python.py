import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# ------------------- [Giữ nguyên] HÀM TÍNH TOÁN -------------------
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Sửa lỗi chia cho 0
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100

    return df

# ------------------- [Giữ nguyên] HÀM GỌI GEMINI CHO PHÂN TÍCH TÓM TẮT -------------------
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# ------------------- [Giữ nguyên] CHỨC NĂNG 1-5 -------------------
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']

        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(
                df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }),
                use_container_width=True
            )

            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            try:
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )

            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"

            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Xử lý để tránh lỗi nếu không tìm thấy chỉ tiêu
            tsnh_growth = "N/A"
            tsnh_row = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]
            if not tsnh_row.empty:
                tsnh_growth = f"{tsnh_row['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%"
            
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    tsnh_growth, 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False)

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY")
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# =================================================================
# ===================  KHUNG CHAT VỚI GEMINI  =====================
# =================================================================

st.divider()
st.header("💬 Khung Chat Gemini (Hỏi–Đáp thời gian thực)")

# Tuỳ chọn model & system prompt
with st.expander("⚙️ Tuỳ chọn nâng cao", expanded=False):
    model_name = st.selectbox(
        "Chọn model Gemini",
        options=[
            "gemini-2.5-flash",
            "gemini-2.0-pro-exp", # Giữ các tùy chọn này để người dùng có thể thử các model khác
            "gemini-2.0-flash-thinking-exp" 
        ],
        index=0,
        key="chat_model_select"
    )
    system_instruction = st.text_area(
        "System instruction (ngữ cảnh vai trò/trợ lý)",
        value=(
            "Bạn là trợ lý AI chuyên nghiệp về tài chính – kế toán – kiểm toán. "
            "Trả lời ngắn gọn, có cấu trúc, kèm công thức/mẹo nếu cần."
        ),
        key="chat_system_instruction"
    )

# Lưu lịch sử hội thoại
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Xin chào! Bạn muốn hỏi gì về báo cáo tài chính, IFRS, phân tích chỉ số…?"}
    ]

def _streamlit_render_messages():
    """Hiển thị lịch sử chat trong giao diện Streamlit."""
    for msg in st.session_state.chat_messages:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
            st.markdown(msg["content"])

def _to_gemini_history(messages, system_instruction_text):
    """
    Chuyển lịch sử hội thoại của Streamlit sang định dạng contents cho Google GenAI.
    Thêm system_instruction dưới dạng một tin nhắn user đặc biệt ở đầu để cung cấp ngữ cảnh.
    """
    contents = []
    # Thêm System Instruction dưới dạng tin nhắn đầu tiên của user (vai trò của mô hình)
    if system_instruction_text and system_instruction_text.strip():
        contents.append({"role": "user", "parts": [{"text": f"[System Instruction]\n{system_instruction_text.strip()}"}]})
    
    # Thêm lịch sử tin nhắn thực tế
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        # Bỏ qua tin nhắn chào mừng ban đầu của assistant khi chuyển đổi
        if m["role"] == "assistant" and m["content"].startswith("Xin chào!"):
            continue
            
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
        
    return contents

# Render lịch sử
_streamlit_render_messages()

# Ô chat input
user_input = st.chat_input("Nhập câu hỏi cho Gemini…")
if user_input:
    # 1. Thêm tin nhắn người dùng vào lịch sử và hiển thị
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        with st.chat_message("assistant"):
            st.error("Chưa cấu hình GEMINI_API_KEY trong st.secrets. Vui lòng kiểm tra lại.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            # Chuyển đổi lịch sử chat
            contents = _to_gemini_history(st.session_state.chat_messages, system_instruction)
            
            # 2. Gọi API và hiển thị phản hồi
            with st.chat_message("assistant"):
                with st.spinner("Gemini đang soạn trả lời…"):
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=contents
                    )
                    # Lấy nội dung hoặc thông báo lỗi nếu có
                    answer = getattr(resp, "text", None) or "Không nhận được nội dung từ mô hình."
                    st.markdown(answer)
                    # 3. Lưu phản hồi của AI vào lịch sử
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

        except APIError as e:
            with st.chat_message("assistant"):
                st.error(f"Lỗi gọi Gemini API: {e}. Vui lòng kiểm tra Khóa API.")
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Đã xảy ra lỗi không xác định: {e}")
        # Buộc rerun để cập nhật khung chat ngay lập tức
        st.rerun()

# Nút xoá lịch sử chat
col_reset, _ = st.columns([1, 5])
with col_reset:
    if st.button("🧹 Xoá lịch sử chat"):
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Lịch sử đã được xoá. Bạn cần hỏi gì, cứ nhắn mình nhé!"}
        ]
        st.rerun()
