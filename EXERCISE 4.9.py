#Exercise 4.9
# Modify Example 4.10 so that it loads the VGG19 model instead of VGG16.
from keras.applications.vgg19 import VGG19

model = VGG19(weights='imagenet')
model.summary()
# VGG19 has 3 more conv layers than VGG16 (one more in blocks 3, 4 and 5),
# about 143.7M parameters vs 138.4M.
