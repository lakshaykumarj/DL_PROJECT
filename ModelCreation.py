import os
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

cifar_10_dir = "C:\\Users\\LAKSHAY KUMAR\\Downloads\\cifar-10-python\\cifar-10-batches-py"
batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

all_data = []
all_labels = []
for batch_file in batch_files:
    file_path = os.path.join(cifar_10_dir, batch_file)
    data_dict = unpickle(file_path)
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    all_data.append(data)
    all_labels += labels

data = np.vstack(all_data).reshape(-1, 3, 32, 32)
data = data.transpose(0, 2, 3, 1)
labels = np.array(all_labels)

label_names_file = os.path.join(cifar_10_dir, 'batches.meta')
label_names_dict = unpickle(label_names_file)
label_names = label_names_dict[b'label_names']

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 1))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(data[i])
    plt.title(label_names[labels[i]])
    plt.axis('off')
plt.show()


num_images_per_label = 5
displayed_labels = set()
for i in range(len(data)):
    label = label_names[labels[i]]
    if label not in displayed_labels:
        label_indices = np.where(labels == labels[i])[0][:num_images_per_label]
        plt.figure(figsize=(10, 2))
        for j, idx in enumerate(label_indices):
            plt.subplot(1, num_images_per_label, j + 1)
            plt.imshow(data[idx])
            plt.title(f"{label_names[labels[idx]]}")
            plt.axis('off')
        plt.show()
        displayed_labels.add(label)
    if len(displayed_labels) == 10:
        break


import tensorflow as tf
from tensorflow import keras
data = data / 255.0
# Define the CNN model
model = keras.Sequential([
    
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history  = model.fit(data, labels, epochs=20, validation_split=0.2)

import random
# Load your custom CIFAR-10 dataset
cifar_10_dir = "C:\\Users\\LAKSHAY KUMAR\\Downloads\\cifar-10-python\\cifar-10-batches-py"
batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
all_data = []
all_labels = []
for batch_file in batch_files:
    file_path = os.path.join(cifar_10_dir, batch_file)
    data_dict = unpickle(file_path)
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    all_data.append(data)
    all_labels += labels
data = np.vstack(all_data).reshape(-1, 3, 32, 32)
data = data.transpose(0, 2, 3, 1)
labels = np.array(all_labels)
label_names_file = os.path.join(cifar_10_dir, 'batches.meta')
label_names_dict = unpickle(label_names_file)
label_names = label_names_dict[b'label_names']

random_indices = random.sample(range(len(data)), 10)

def plot_image_with_rectangle(image, label, predicted_label, match):
    plt.imshow(image)
    title = f"Actual: {label}\nPredicted: {predicted_label}"
    plt.title(title, color='green' if match else 'red')
    plt.axis('off')

plt.figure(figsize=(16, 3))
for i, idx in enumerate(random_indices):
    plt.subplot(1, 11, i + 1)
    image = data[idx]
    label = label_names[labels[idx]].decode("utf-8")
    predicted_label = model.predict(np.expand_dims(image/255.0, axis=0))  
    predicted_class_index = np.argmax(predicted_label)
    predicted_label = label_names[predicted_class_index].decode("utf-8")
    match = label == predicted_label
    plot_image_with_rectangle(image, label, predicted_label, match)
    rect_color = 'green' if match else 'red'
    plt.gca().add_patch(plt.Rectangle((0, 0), 32, 32, linewidth=2, edgecolor=rect_color, facecolor='none'))
plt.tight_layout()
plt.show()

