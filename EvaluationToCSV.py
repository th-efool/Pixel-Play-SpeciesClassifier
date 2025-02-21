import tensorflow as tf
import os
import pandas as pd
from keras.utils import load_img, img_to_array
import numpy as np

# Constants
IMAGE_SIZE = 224
testDirectory = "C:/Projects Temp/Pixel Play/vlg-recruitment-24-challenge/vlg-dataset/test"

# Dataset and Model Paths
DatasetPath = "C:/Projects Temp/Pixel Play/dataset"
trainPath = DatasetPath + "/train"
best_model_file = "C:/Projects Temp/Pixel Play/temp/inception_resnet_finetunedA.keras"

# Load Model
CLASSES = sorted(os.listdir(trainPath))  # Sorting ensures consistent ordering of class indices
model = tf.keras.models.load_model(best_model_file)

print(model.summary())

# Image Preparation Function
def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.0
    return imgResult

# Process All Test Images
results = []
for file_name in os.listdir(testDirectory):
    file_path = os.path.join(testDirectory, file_name)
    if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.png')):
        imgForModel = prepareImage(file_path)
        resultArray = model.predict(imgForModel, verbose=0)  # Silent prediction
        predicted_index = np.argmax(resultArray, axis=1)[0]
        predicted_class = CLASSES[predicted_index]
        results.append({'image_id': file_name, 'class': predicted_class})

# Save Results to CSV
output_csv_path = "C:/Projects Temp/Pixel Play/LombWob.csv"
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
