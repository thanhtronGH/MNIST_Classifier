import myRFLibs
import numpy as np
import RF_Model as model
import streamlit as st
from PIL import Image, ImageOps


# Init node (Bắt buộc phải giữ lại ở đây để pickle load không bị lỗi __main__)
class Node:

    def __init__(
        self,
        feature_idx=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


# Load 99% accuracy model
forestNumber = model.load_model("forestNumber.pkl")

st.title("MNIST Handwritten Digit Classifier")

# Load MNIST image
MINISTImg = st.file_uploader(
    "Drag an image here", type=["png", "jpg", "jpeg"]
)

# Tạo một biến toàn cục để chứa ảnh đã xử lý làm phẳng
img_flatten = None

# CHỈ XỬ LÝ KHI NGƯỜI DÙNG ĐÃ TẢI ẢNH LÊN
if MINISTImg is not None:
    # 1. Mở ảnh gốc bằng PIL và hiển thị lên giao diện cho người dùng xem
    image = Image.open(MINISTImg)
    st.image(image, caption="Uploaded image", width=150)

    # 2. TIỀN XỬ LÝ ẢNH CHUẨN MNIST (Quan trọng để mô hình chạy được):
    # - Chuyển sang ảnh xám (Grayscale)
    img_gray = ImageOps.grayscale(image)
    # - Thay đổi kích thước (Resize) về đúng định dạng 28x28 pixel
    img_resized = img_gray.resize((28, 28))

    # 3. CHUYỂN ĐỔI SANG MẢNG NUMPY VÀ LÀM PHẲNG
    img_np = np.array(img_resized, dtype=np.float32)

    # LƯU Ý MẸO: Tập MNIST chuẩn là nền đen (0) chữ trắng (255).
    # Nếu ảnh bạn tải lên là chữ đen nền trắng, hãy bỏ dấu thăng '#' dòng dưới để đảo ngược màu:
    # img_np = 255.0 - img_np

    # Làm phẳng ma trận 28x28 thành mảng 1 chiều 784 phần tử
    img_flatten_raw = img_np.flatten()

    # Thêm một chiều để biến thành ma trận dạng (1, 784) phù hợp cho hàm dự đoán vòng lặp
    img_flatten = np.expand_dims(img_flatten_raw, axis=0)

else:
    # Nếu chưa tải ảnh, xóa kết quả cũ trong session_state để tránh hiển thị sai
    if "result" in st.session_state:
        del st.session_state.result
    st.info("Please upload an image of a handwritten digit to proceed.")

# NÚT BẤM DỰ ĐOÁN
if st.button("Predict"):
    if img_flatten is not None:
        # Gọi hàm dự đoán từ thư viện của bạn với ma trận (1, 784)
        result = myRFLibs.predict_forest(forestNumber, img_flatten)

        # Lấy giá trị dự đoán (nếu hàm trả về một mảng/list kết quả, ta lấy phần tử đầu tiên)
        if isinstance(result, (list, np.ndarray)):
            st.session_state.result = result[0]
        else:
            st.session_state.result = result
    else:
        st.error("Please upload an image before clicking Predict.")

# HIỂN THỊ KẾT QUẢ
if "result" in st.session_state:
    st.header(f"A Predicted number is: :green[{st.session_state.result}]")

st.divider()
