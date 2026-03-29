import numpy as np
import cv2

# Load image
image = cv2.imread('digits1.png')

if image is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Split image into 50 rows and 100 columns (20x20 each)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray_img, 50)]

# Convert to numpy array
x = np.array(cells)

# Prepare train and test data
train_data = x[:, :50].reshape(-1, 400).astype(np.float32)
test_data = x[:, 50:100].reshape(-1, 400).astype(np.float32)

# Create labels (0-9 digits)
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = np.repeat(k, 250)[:, np.newaxis]

# Train KNN model
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Predict
ret, result, neighbours, dist = knn.findNearest(test_data, k=3)

# Accuracy check
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size

print("Accuracy:", accuracy)