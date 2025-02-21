import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# Paths
DatasetPath = "C:/Projects Temp/Pixel Play/dataset"
trainPath = DatasetPath + "/train"
validPath = DatasetPath + "/validate"
testPath = DatasetPath + "/test"
best_model_file = "C:/Projects Temp/Pixel Play/temp/model42-InceptRestV2.keras"  # Pretrained model location
finetuned_model_file = "C:/Projects Temp/Pixel Play/temp/inception_resnet_finetunedA.keras"

# Number of classes
NumberOfSpecies = len(os.listdir(trainPath))  # Ensure this matches the number of classes in your dataset

# Data Generators with augmentations for train and rescaling for validation/test
train_data = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    zoom_range=0.2,  # Increased augmentation to improve generalization
    rotation_range=20
).flow_from_directory(
    directory=trainPath,
    batch_size=32,
    target_size=(224, 224),
    class_mode="categorical"
)

valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255
).flow_from_directory(
    directory=validPath,
    batch_size=32,
    target_size=(224, 224),
    class_mode="categorical"
)

# Load previously saved best model
print("Loading the pre-trained model...")
model = load_model(best_model_file)

# Freezing the base model layers and ensure fine-tuning the dense layer/head
for layer in model.layers[:-1]:  # Freeze all layers except the final Dense layer
    layer.trainable = False

# Ensuring the final Dense layer is trainable (for fine-tuning classification)
model.layers[-1].trainable = True

# Recompiling the model with a low learning rate for fine-tuning
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    metrics=["accuracy"]
)

# Callbacks for training
callbacks = [
    ModelCheckpoint(
        filepath=finetuned_model_file,  # Save weights for the best model
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.1,
        patience=3,  # Reduce LR if validation accuracy plateaus for 3 epochs
        verbose=1,
        min_lr=0.000001
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,  # Stop training if no improvement for 5 epochs
        verbose=1,
        restore_best_weights=True
    )
]

# Fine-tune the model
EPOCHS = 20  # Fine-tune for fewer epochs as weights are already pretrained
print("Fine-tuning the model...")
history = model.fit(
    train_data,
    epochs=EPOCHS,
    steps_per_epoch=len(train_data),
    validation_data=valid_data,
    validation_steps=int(0.25 * len(valid_data)),  # Use only 25% validation to save time
    callbacks=callbacks
)

# Evaluate the fine-tuned model on the test set
print("Evaluating the model on the test set...")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# Plot loss and accuracy curves
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


plot_loss_curves(history)
