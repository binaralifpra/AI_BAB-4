#Example 4.6
from keras.applications import inception_v3

# init the models
model = inception_v3.InceptionV3(weights='imagenet')
model.summary()
