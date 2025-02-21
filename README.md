## Overview

This project implements a deep learning model for multi-class image classification using TensorFlow and Keras. The training was conducted in two distinct phases with an efficient combination of layer structuring and training strategies.
Builded up dataset for the unseen classes using “Custom Search JSON API” and using bing through “beautifulsoup” 

### Training Phases

#### Phase 1: Aggressive Training with High Learning Rate

- **Learning Rate:** Adam optimizer with a learning rate of 0.01
- **Training Epochs:** 8 epochs
- **Layer Structure:**
    - Pretrained "InceptionResNetV2" model
    - Global Average Pooling 2D layer
    - Dense output layer (activation="softmax")

#### Phase 2: Fine-Tuning with Lower Learning Rate and Dropout Layer
- **Learning Rate:** Adam optimizer with a learning rate of 0.000001
- **Model Initialization:** Loaded the best model from Phase 1
- **Dropout:** Added a `Dropout(0.15)` layer to prevent overfitting
- **Dense Layers:** Final Dense output layer with softmax activation.
- Other configurations with the follow  did not improve performance significantly.
     - additional ReLU-activated dense layers
     - trying unfreezing top30 layers, top100, last 30
     - Applying weights to attended features using "Multiply()([x, attention_weights])" before procedding to concatenating GAP2D GMP2D
     - Other attempts to introduce some non-linearity

### Dataset Details

- **Path:** `C:/Projects Temp/Pixel Play/dataset`
    - **Train Directory:** `/train`
    - **Validation Directory:** `/validate`
- **Number of Classes:** `NumberOfSpecies` (calculated as the count of directories in the training folder)

---

## Code Highlights

### Preprocessing Pipelines

#### Data Generators

- **Training Data:** Applied horizontal flip, zoom range, rotation, and rescaling.
- **Validation & Test Data:** Only applied rescaling.

#### Code Snippet

```python
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=15
)
```

### Model Architecture

- Used the pretrained `InceptionResNetV2` as the base, frozen initially.
- Included custom classification layers tailored to the number of species.

#### Key Model Components

```python
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.15)(x)  # 15% Dropout
outputs = tf.keras.layers.Dense(NumberOfSpecies, activation="softmax", name="Output_Layer")(x)
model = tf.keras.Model(inputs, outputs)
```

### Training Configuration

- **Optimizer:** Adam with adaptive learning rate
- **Loss Function:** Categorical crossentropy
- **Metrics:** Accuracy

#### Callbacks Used

1. **ModelCheckpoint:** Save the best model based on validation accuracy.
2. **ReduceLROnPlateau:** Reduce the learning rate on a plateau.
3. **EarlyStopping:** Stop training early if performance stagnates.

### Evaluation & Plotting

- **Performance Evaluation:** The model’s accuracy was validated on unseen test data.
- **Loss & Accuracy Visualization:** Training and validation performance plotted for analysis.

```python
def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(loss))

    # Plot Loss
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.legend()
    plt.show()
```
