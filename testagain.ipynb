
from IPython.display import display
from google.colab.output import eval_js
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from base64 import b64decode

# Load model (update path as needed)
model = tf.keras.models.load_model('mask_detector.keras')

# Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# JS code as string
js_code = '''
async function capture(){
  const div = document.createElement('div');
  const video = document.createElement('video');
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  const stream = await navigator.mediaDevices.getUserMedia({video:true});
  div.appendChild(video);
  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  stream.getTracks().forEach(track => track.stop());
  return canvas.toDataURL('image/jpeg', 0.8);
}
capture();
'''

# Run JS and get base64 image string
data = eval_js(js_code)

# Decode base64 string to image
img_data = b64decode(data.split(',')[1])
nparr = np.frombuffer(img_data, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Detect faces and classify
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

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
