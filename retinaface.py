# -*- coding: utf-8 -*-
# RetinaFace.py

# Install the retina-face library if not installed
# Run this command once, and comment it out after installation
# !pip install retina-face 

# Import Retinaface, CV2, and Matplotlib
from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

# If you are using Jupyter, you can use IPython.display to display images
# from IPython.display import display

# Read and display an image
img_path = 'CSK-with-Trophy.jpeg'  # Replace this with your image file path
img = cv2.imread(img_path)

# If running locally, use cv2.imshow
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect faces using RetinaFace
img_faces = RetinaFace.detect_faces(img)

# Iterate over all detected face key areas
for i in img_faces.keys():
    choose = img_faces[i]
    print(choose)

# Iterate all objects key areas and mark only face area
for i in img_faces.keys():
    facial_parts = img_faces[i]
    recognize_face_area = facial_parts["facial_area"]
    cv2.rectangle(img, (recognize_face_area[2], recognize_face_area[3]),
                  (recognize_face_area[0], recognize_face_area[1]), (255, 255, 255), 1)

# Display Face Area
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Extract individual faces from the image
ext_faces = RetinaFace.extract_faces(img_path=img_path, align=True)
for face in ext_faces:
    plt.imshow(face)
    plt.show()
