import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
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
