from google.colab import files
files.upload()
# Step 1: Upload the zip file
import zipfile
import os
zip_files_in_current_dir = [f for f in os.listdir('.') if f.endswith('.zip')]

if zip_files_in_current_dir:
    zip_filename = zip_files_in_current_dir[0]
    print(f"Detected uploaded zip file: {zip_filename}")
else:
    print("Error: No zip files found in the current directory. Please ensure a zip file was uploaded correctly.")
    zip_filename = None 
    # Step 2: Extract the zip file
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('flowers_dataset')
print("Extraction completed.")
# Step 2: Extract the zip file
!ls flowers_dataset
# Step 3: Check extracted directory
import tensorflow as tf
# Step 4: Load data using TensorFlow's image data generator
# Define path to extracted dataset
dataset_path = 'flowers_dataset'
# You can adjust image size and batch size
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
# Create training and validation datasets from the directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)
    # Step 5: Visualize some training images 
import matplotlib.pyplot as plt

class_names = train_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
# Step 6: Normalize the data (scale pixel values)
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# Step 7: Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# Step 8: Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=IMG_SIZE + (3,)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
# Step 9: Train the model
EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)

plt.show()
# ==== Step 9: Confusion matrix and classification report ====

# Predict classes on test data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


test_generator = test_datagen.flow_from_directory(
    dataset_path, # points to 'flowers_dataset'
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse', # for integer labels, compatible with sparse_categorical_crossentropy
    shuffle=False,
    subset='validation' # This creates a generator for the validation subset of 'flowers_dataset'
)

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get true labels
y_true = test_generator.classes

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
from sklearn.metrics import classification_report

# Classification report for precision, recall, f1-score
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=class_names))
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2,2)),

        # Second convolutional block
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),

        # Third convolutional block
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),

        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),          # helps prevent overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        # Output layer with softmax activation for multi-class classification
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Example usage:
# input shape (height, width, channels), e.g. (180, 180, 3) for RGB images resized to 180x180
input_shape = (180, 180, 3)
num_classes = 5  # replace with actual number of flower classes

model = create_cnn_model(input_shape, num_classes)
model.summary()
