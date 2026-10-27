#Example 4.11
# Needs an image named Elephant.jpg in the same folder as this script.
import os
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

model = VGG16(weights='imagenet')

img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# prediction
predictions = model.predict(x)
results = decode_predictions(predictions)
print(results)
