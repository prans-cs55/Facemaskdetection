import numpy as np
import tensorflow as tf
from google.colab import files
import matplotlib.pyplot as plt

# Upload your image file
uploaded = files.upload()
image_path = list(uploaded.keys())[0]  # Get uploaded file name

# Load your trained model (update the path if needed)
model = tf.keras.models.load_model('mask_detector.keras')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the uploaded image
img = cv2.imread(image_path)
if img is None:
    print("Could not read the image.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    prediction = model.predict(face_img)[0][0]
    label = "Mask" if prediction < 0.5 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 6)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)

# Convert BGR to RGB for displaying correctly with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show the image inline
plt.figure(figsize=(10,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
