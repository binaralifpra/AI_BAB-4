#Example 4.12
# Needs an image named Elephant.jpg in the same folder as this script.
# Uncomment one model and its matching preprocess_input to switch networks.
# Note: InceptionV3 expects 299x299 images, so change target_size for it.
# The book passes x instead of processed_image to predict(); fixed here.
import os
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

# init the models
#model = vgg16.VGG16(weights='imagenet')
#model = resnet50.ResNet50(weights='imagenet')
model = mobilenet.MobileNet(weights='imagenet')
#model = inception_v3.InceptionV3(weights='imagenet')
model.summary()

img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

#processed_image = vgg16.preprocess_input(x)
#processed_image = resnet50.preprocess_input(x)
processed_image = mobilenet.preprocess_input(x)
#processed_image = inception_v3.preprocess_input(x)

# prediction
predictions = model.predict(processed_image)
results = decode_predictions(predictions)
print(results)
