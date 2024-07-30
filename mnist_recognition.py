# mnist_recognition.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Print the shape of the data
print(f'Training data shape: {train_images.shape}')
print(f'Test data shape: {test_images.shape}')

# Preprocess the data
# Normalize the images to the range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Print the shape of the preprocessed data
print(f'Preprocessed training data shape: {train_images.shape}')
print(f'Preprocessed test data shape: {test_images.shape}')
print(f'Preprocessed training labels shape: {train_labels.shape}')
print(f'Preprocessed test labels shape: {test_labels.shape}')


# mnist_recognition.py

# Function to display images
def display_images(images, labels, num_images):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(np.argmax(labels[i]))
        plt.axis('off')
    plt.show()

# Display the first 5 images and their labels
display_images(train_images, train_labels, 5)
