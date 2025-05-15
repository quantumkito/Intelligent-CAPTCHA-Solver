from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 10,
    zoom_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    fill_mode = 'nearest',
    horizontal_flip = True
)

test_datagen = ImageDataGenerator()

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(10, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001), 
    metrics=['accuracy']
)

def model_evaluation(dataX, dataY, k, epochs):
    print('[Model Evaluation]')

    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    
    fold = 0  
    accuracies = []  
    losses = []  

    for train_ix, test_ix in kfold.split(dataX, np.argmax(dataY, axis=1)):  
        fold += 1
        print(f'Fold: {fold}/{k}')
        
        X_train, Y_train = dataX[train_ix], dataY[train_ix]
        X_test, Y_test = dataX[test_ix], dataY[test_ix]

        train_datagen.fit(X_train)
        test_datagen.fit(X_test)

        train_iterator = train_datagen.flow(X_train, Y_train, batch_size=32)
        test_iterator = test_datagen.flow(X_test, Y_test, batch_size=32)

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(train_iterator, epochs=epochs, validation_data=test_iterator, callbacks=[early_stop])

        loss, accuracy = model.evaluate(X_test, Y_test)
        losses.append(loss)
        accuracies.append(accuracy)

    return np.mean(losses), np.mean(accuracies)