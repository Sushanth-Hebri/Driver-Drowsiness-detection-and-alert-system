import os
import cv2
import numpy as np
import tensorflow as tf

# Load your model using TensorFlow's Keras API
model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\Miniproject2\drowiness_new7.h5')

# Function to load and preprocess images
def load_images_from_folder(folder, label, img_size):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels

# Set paths to the folders
drowsy_path = r'C:\Users\HP\Downloads\ddds\Drowsy'
non_drowsy_path = r'C:\Users\HP\Downloads\ddds\Non Drowsy'
img_size = (64, 64)  # Use the input size expected by your model

# Load images
drowsy_images, drowsy_labels = load_images_from_folder(drowsy_path, 1, img_size)
non_drowsy_images, non_drowsy_labels = load_images_from_folder(non_drowsy_path, 0, img_size)

# Combine and prepare data
X_test = np.array(drowsy_images + non_drowsy_images)
y_test = np.array(drowsy_labels + non_drowsy_labels)

# Normalize the images if your model expects normalized input
X_test = X_test / 255.0

# Make predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
