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

# --- Khởi tạo Session State cho Chat ---
# Lưu trữ lịch sử tin nhắn
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
# Lưu trữ đối tượng chat session của Gemini để duy trì ngữ cảnh
if 'chat_session' not in st.session_state:
    st.session_state['chat_session'] = None
# Cờ đánh dấu đã hoàn thành phân tích ban đầu và có thể chat
if 'analysis_completed' not in st.session_state:
    st.session_state['analysis_completed'] = False

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN XỬ LÝ TỶ TRỌNG *******************************
    # Xử lý trường hợp Tổng Tài sản bằng 0 để tránh lỗi chia cho 0
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* KẾT THÚC PHẦN XỬ LÝ TỶ TRỌNG *******************************
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Ban đầu và Khởi tạo Chat ---
def init_gemini_chat_and_analyze(data_for_ai, api_key):
    """
    1. Khởi tạo chat session với ngữ cảnh (context) là dữ liệu tài chính.
    2. Gửi tin nhắn đầu tiên để lấy nhận xét ban đầu.
    3. Lưu chat session và tin nhắn vào session state.
    """
    
    # System Instruction: Thiết lập vai trò và ngữ cảnh (được dùng làm bối cảnh cho chat session)
    system_instruction = f"""
    Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Mọi câu trả lời của bạn phải dựa trên dữ liệu tài chính được cung cấp dưới đây. Hãy đảm bảo giữ ngữ cảnh của dữ liệu này trong suốt cuộc trò chuyện.
    Dữ liệu thô và chỉ số:
    {data_for_ai}
    """
    
    # Prompt đầu tiên: Yêu cầu phân tích tổng quan
    initial_prompt = """
    Dựa trên dữ liệu tài chính trong ngữ cảnh hệ thống, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
    """
    
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        
        # 1. Khởi tạo Chat Session với System Instruction
        chat = client.chats.create(
            model=model_name,
            system_instruction=system_instruction
        )
        
        # 2. Gửi tin nhắn đầu tiên để lấy nhận xét tổng quan
        response = chat.send_message(initial_prompt)
        
        # 3. Cập nhật Session State
        st.session_state['chat_session'] = chat
        st.session_state['messages'] = [
            {"role": "user", "content": initial_prompt},
            {"role": "model", "content": response.text}
        ]
        st.session_state['analysis_completed'] = True
        
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm xử lý khung Chat ---
def handle_chat_input():
    """Xử lý đầu vào chat từ người dùng và gửi đến Gemini."""
    user_query = st.session_state.get("chat_input_key")
    
    if user_query and user_query.strip():
        # Thêm tin nhắn người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Gửi tin nhắn và chờ phản hồi
        with st.spinner(f"Đang phân tích câu hỏi: **'{user_query}'**..."):
            try:
                chat = st.session_state.chat_session
                # Gửi tin nhắn qua chat session đã khởi tạo
                response = chat.send_message(user_query)
                
                # Thêm tin nhắn AI vào lịch sử
                st.session_state.messages.append({"role": "model", "content": response.text})
            
            except Exception as e:
                error_message = f"Lỗi khi gửi tin nhắn đến Gemini: {e}"
                st.session_state.messages.append({"role": "model", "content": error_message})
        
        # Xóa nội dung input và buộc Streamlit reruns
        st.session_state["chat_input_key"] = ""
        st.rerun() # Buộc rerun để cập nhật khung chat

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Reset trạng thái nếu tải file mới
if uploaded_file is not None:
    # Nếu file mới được tải lên, reset session state để phân tích lại
    if st.session_state.analysis_completed:
        st.session_state['messages'] = []
        st.session_state['chat_session'] = None
        st.session_state['analysis_completed'] = False

    try:
        # Tải file và tiền xử lý
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khối tính toán chỉ số (giữ nguyên)
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
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
                 thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                 thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                 st.warning("Lỗi chia cho 0: Nợ Ngắn Hạn năm hiện tại hoặc năm trước bằng 0. Không thể tính Chỉ số Thanh toán Hiện hành.")
                 thanh_toan_hien_hanh_N = "N/A" 
                 thanh_toan_hien_hanh_N_1 = "N/A"


            # Chuẩn bị dữ liệu để gửi cho AI 
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    (f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%"
                     if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else "N/A"),
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            # --- Chức năng 5: Khởi tạo Chat AI và Khung Chat Hỏi Đáp ---
            st.subheader("5. Hỏi đáp chuyên sâu với Gemini AI")
            
            api_key = st.secrets.get("GEMINI_API_KEY") 
            
            if not api_key:
                st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
            
            elif not st.session_state.analysis_completed:
                # Nút nhấn để bắt đầu phân tích và khởi tạo Chat
                if st.button("Yêu cầu AI Phân tích Tổng quan & Bắt đầu Chat"):
                    with st.spinner('Đang khởi tạo chat session và chờ Gemini phân tích...'):
                        ai_result = init_gemini_chat_and_analyze(data_for_ai, api_key)
                        
                        # Chỉ hiển thị kết quả phân tích ban đầu nếu thành công
                        if "Lỗi" not in ai_result:
                            st.markdown("**Kết quả Phân tích Ban đầu từ Gemini AI:**")
                            st.info(ai_result)
                        else:
                            st.error(ai_result)
                            # Nếu lỗi, đảm bảo chat_session không bị set
                            st.session_state['analysis_completed'] = False
                        # Sau khi khởi tạo, Streamlit sẽ rerun và chuyển sang chế độ Chat
                        st.rerun() 
            
            # --- Khung Chat Tương tác ---
            if st.session_state.analysis_completed:
                st.markdown("**Bắt đầu hỏi đáp:** Bạn có thể hỏi thêm về bất kỳ chỉ tiêu hoặc xu hướng nào trong dữ liệu trên. Ví dụ: *'Đánh giá chi tiết hơn về tốc độ tăng trưởng của Tài sản ngắn hạn.'*")
                
                # Hiển thị tất cả tin nhắn
                for message in st.session_state.messages:
                    # Lọc tin nhắn đầu tiên của user (là prompt yêu cầu phân tích) để không hiển thị trong khung chat
                    if len(st.session_state.messages) == 2 and message["role"] == "user":
                        continue 
                        
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Khung nhập Chat
                st.chat_input(
                    "Đặt câu hỏi của bạn về dữ liệu tài chính...", 
                    key="chat_input_key", 
                    on_submit=handle_chat_input
                )
            
    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
