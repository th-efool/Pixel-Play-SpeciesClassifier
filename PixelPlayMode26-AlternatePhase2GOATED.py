import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Paths
DatasetPath = "C:/Projects Temp/Pixel Play/Splitted Up"
trainPath = DatasetPath + "/train"
validPath = DatasetPath + "/validate"
testPath = DatasetPath + "/test"
best_model_file = "C:/Projects Temp/Pixel Play/temp/XXVInceptionResNetV2.keras"
finetuned_model_file = "C:/Projects Temp/Pixel Play/temp/inception_resnet_finetunedX.keras"

# Number of species (classes)
NumberOfSpecies = len(os.listdir(trainPath))

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=15
)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    directory=trainPath,
    batch_size=32,
    target_size=(224, 224),
    class_mode="categorical"
)
valid_data = valid_datagen.flow_from_directory(
    directory=validPath,
    batch_size=32,
    target_size=(224, 224),
    class_mode="categorical"
)
test_data = test_datagen.flow_from_directory(
    directory=testPath,
    batch_size=32,
    target_size=(224, 224),
    class_mode="categorical"
)

# Load the Best Model
base_model = load_model(best_model_file)
print("Best model loaded successfully.")

# Add a Dropout Layer and a New Classifier
# Get the base model's output
base_model_output = base_model.output

# First Dropout and ReLu Activated Dense Layers
x =  tf.keras.layers.Dropout(0.25)(base_model_output)  # 25% Dropout
x =  tf.keras.layers.Dense(512, activation="relu", name="Intermediate_Dense_Layer")(x)  # Intermediate Dense Layer

# Second Dropout and Final Dense Layers
x =  tf.keras.layers.Dropout(0.2)(x)  # 20% Dropout
x =  tf.keras.layers.Dense(NumberOfSpecies, activation="softmax", name="Output_Layer")(x)  # Final Classification Layer

# Create the Fine-Tuned Model
model = tf.keras.Model(inputs=base_model.input, outputs=x)


# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the Model
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    metrics=["accuracy"]
)
print("Model ready for fine-tuning with Dropout.")

# Callbacks
callbacks = [
    ModelCheckpoint(finetuned_model_file, save_best_only=True, monitor="val_accuracy", verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=3, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
]

# Fine-Tune the Model
EPOCHS = 10
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    steps_per_epoch=len(train_data),
    validation_steps=len(valid_data),
    callbacks=callbacks
)

# Evaluate the Fine-Tuned Model
print("Validation Evaluation:")
val_results = model.evaluate(valid_data)

print("\nTest Evaluation:")
test_results = model.evaluate(test_data)

# Plot Loss and Accuracy Curves
import matplotlib.pyplot as plt

def plot_loss_curves(history):
    # Extract metrics
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(loss))

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_loss_curves(history)
