from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy
import matplotlib.pyplot as plt

model = load_model()

image = load_img('example_image.png', color_mode='grayscale', target_size=(28, 28))
image = img_to_array(image)
image = image.reshape(-1, 28, 28, 1)
image = image / 255.0

plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title("Preprocessed Image")
plt.show()

prediction = model.predict(image)[0]
predicted_label = numpy.argmax(prediction)

print('Predicted Digit:', predicted_label)