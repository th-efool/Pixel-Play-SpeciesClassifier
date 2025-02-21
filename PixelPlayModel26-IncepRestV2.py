import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.image as mping
# from PIL import Image, UnidentifiedImageError  # For image validation

DatasetPath = "C:/Projects Temp/Pixel Play/dataset"
trainPath = DatasetPath + "/train"
validPath = DatasetPath + "/validate"

NumberOfSpecies = len(os.listdir(trainPath))
best_model_file = "C:/Projects Temp/Pixel Play/temp/model42-InceptRestV2.keras"

# Rescale the image -> from 0-255 to 0-1
train_data = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=15
).flow_from_directory(
    directory=testPath,
    batch_size=32,
    target_size=(224, 224),
    class_mode="categorical"
)

valid_data = ImageDataGenerator(
    rescale=1.0 / 255
).flow_from_directory(
    directory=validPath,
    batch_size=32,
    target_size=(224, 224),
    class_mode="categorical"
)

# Load EfficientNetB3
base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze all layers initially

# Add custom classification layers
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x) #GAP2D Layer
x =  tf.keras.layers.Dropout(0.15)(x)  # 15% Dropout
outputs = tf.keras.layers.Dense(NumberOfSpecies, activation="softmax", name="Output_Layer")(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
)

# Callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=4, verbose=1, min_lr=0.000001),
    EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
]

print(model.summary())

# Train the model
EPOCHS = 30
history = model.fit(
    train_data,
    epochs=EPOCHS,
    steps_per_epoch=len(train_data),
    validation_data=valid_data,
    validation_steps=int(0.25 * len(valid_data)),
    callbacks=callbacks
)

# Plot loss curves
def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(history.history["loss"]))

    # Plot the Loss
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # Plot the Accuracy
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


# Run the function
plot_loss_curves(history)
