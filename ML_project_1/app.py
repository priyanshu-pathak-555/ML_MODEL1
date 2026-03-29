import streamlit as st
import numpy as np
import cv2

# Title
st.title("🧠 Handwritten Digit Recognition (kNN)")

# Load and train model
@st.cache_resource
def load_model():
    image = cv2.imread('digits1.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)

    train = x[:, :50].reshape(-1, 400).astype(np.float32)
    test = x[:, 50:100].reshape(-1, 400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]

    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    return knn


knn = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload a digit image (20x20)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (20, 20))
    sample = resized.reshape(-1, 400).astype(np.float32)

    # Prediction
    ret, result, neighbours, dist = knn.findNearest(sample, k=3)

    st.image(resized, caption="Processed Image", width=150)
    st.write(f"### Predicted Digit: {int(result[0][0])}")