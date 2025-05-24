import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)

image_path = "DataSet/images/"
image_labels = sorted(os.listdir(image_path))

img_list = []
label_list = []

for label in image_labels:
    label_dir = os.path.join(image_path, label)
    for img_file in os.listdir(label_dir):
        img_list.append(os.path.join(label_dir, img_file))
        label_list.append(label)

df = pd.DataFrame({'img': img_list, 'label': label_list})
print(df['label'].value_counts())

try:
    print("Sample image shape:", plt.imread(df['img'][0]).shape)
except Exception as e:
    print(f"Error reading sample image: {e}")

df_labels = {label: idx for idx, label in enumerate(image_labels)}  
df['encode_label'] = df['label'].map(df_labels).astype(np.int32)

print(df.head())

datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    shear_range=0.3,
    fill_mode='nearest',
    rotation_range=4
)

X = []
for img_path in df['img']:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  
    X.append(img)

y = df['encode_label']

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, random_state=42)

it_train = datagen.flow(np.array(X_train), y_train.to_numpy(), batch_size=32)
it_val = datagen.flow(np.array(X_val), y_val.to_numpy(), batch_size=32)

base_model = VGG16(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    InputLayer(input_shape=(128, 128, 3)),
    base_model,
    Flatten(),
    Dropout(0.3),  
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.3),  
    Dense(len(df_labels), activation='softmax') 
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

history = model.fit(it_train, epochs=20, validation_data=it_val)
