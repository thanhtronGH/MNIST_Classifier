import streamlit as st
import RF_Model as model
from PIL import Image
import numpy as np
import myRFLibs

# Init node
class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx    
        self.threshold = threshold  
        self.left = left                  
        self.right = right                
        self.value = value                

    def is_leaf(self):
        return self.value is not None
    
# Load 99% accuracy model
forestNumber = model.load_model("forestNumber.pkl")

# Load MINIST image 
MINISTImg = st.file_uploader("Drag an image here", type=["png", "jpg", "jpeg"])
img_np = np.array(MINISTImg, dtype=np.float32)
if MINISTImg is not None:
    image = Image.open(MINISTImg)
    img_flatten = np.array(img_np).squeeze()
    # Hiển thị ảnh lên giao diện
    st.image(image, caption="Uploaded image", use_container_width=True)

if st.button("Predict"):

    result = myRFLibs.predict_forest(forestNumber, img_flatten)

    st.session_state.result = result

# Result
if "result" in st.session_state:
    st.write("A Predicted number is:", st.session_state.result)
    
    
st.divider()

