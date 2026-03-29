
# Title
st.title("🧠 Handwritten Digit Recognition (kNN)")
import streamlit as st
import numpy as np
import cv2
import os

# Title
st.title("🧠 Handwritten Digit Recognition (kNN)")

# Load and train model
@st.cache_resource
def load_model():
    # 1. Get the directory where this app.py file is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Join that directory with the image filename
    image_path = os.path.join(BASE_DIR, 'digits1.png')

    # 3. Check if file exists using the full path
    if not os.path.exists(image_path):
        st.error(f"❌ {image_path} file not found!")
        return None

    # 4. Read the image using the full path
    image = cv2.imread(image_path)

    if image is None:
        st.error("❌ Failed to load image!")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Split into cells
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)

    # Train data
    train = x[:, :50].reshape(-1, 400).astype(np.float32)

    # Labels
    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]

    # Train KNN
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    return knn

# ... (Keep the rest of your code from "# Load model" downwards exactly the same) ...
# Load and train model
# @st.cache_resource
# def load_model():
#     # Check if file exists
#     if not os.path.exists("digits1.png"):
#         st.error("❌ digits1.png file not found!")
#         return None

#     image = cv2.imread('digits1.png')

#     if image is None:
#         st.error("❌ Failed to load image!")
#         return None

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Split into cells
#     cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
#     x = np.array(cells)

#     # Train data
#     train = x[:, :50].reshape(-1, 400).astype(np.float32)

#     # Labels
#     k = np.arange(10)
#     train_labels = np.repeat(k, 250)[:, np.newaxis]

#     # Train KNN
#     knn = cv2.ml.KNearest_create()
#     knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

#     return knn


# Load model
knn = load_model()

# Stop if model not loaded
if knn is None:
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a digit image (20x20)", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        st.error("❌ Invalid image file!")
        st.stop()

    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (20, 20))

    sample = resized.reshape(-1, 400).astype(np.float32)

    # Prediction
    ret, result, neighbours, dist = knn.findNearest(sample, k=3)

    # Display
    st.image(resized, caption="Processed Image", width=150)
    st.success(f"✅ Predicted Digit: {int(result[0][0])}")

