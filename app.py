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

img_flatten = None

if MINISTImg is not None:
    image = Image.open(MINISTImg)
    st.image(image, caption="Uploaded image", width=150)

    # PreProcess
    img_gray = ImageOps.grayscale(image)
    img_resized = img_gray.resize((28, 28))

    img_np = np.array(img_resized, dtype=np.float32)
    img_flatten_raw = img_np.flatten()
    img_flatten = np.expand_dims(img_flatten_raw, axis=0)
else:
    if "result" in st.session_state:
        del st.session_state.result
    st.info("Please upload an image.")

# Predict button
if st.button("Predict"):
    if img_flatten is not None:
        result = myRFLibs.predict_forest(forestNumber, img_flatten)
        if isinstance(result, (list, np.ndarray)):
            st.session_state.result = result[0]
        else:
            st.session_state.result = result
    else:
        st.error("Please upload an image before clicking Predict.")

if "result" in st.session_state:
    st.header(f"A Predicted number is: :green[{st.session_state.result}]")

st.divider()
