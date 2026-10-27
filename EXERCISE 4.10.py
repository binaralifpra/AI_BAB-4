#Exercise 4.10
# Modify Example 4.11 to use VGG19 for classification and compare its
# performance with VGG16.
# Needs an image named Elephant.jpg in the same folder as this script.
import os
import time
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications import vgg16, vgg19

img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
img = load_img(img_path, target_size=(224, 224))
x = np.expand_dims(img_to_array(img), axis=0)

for name, module, cls in [("VGG16", vgg16, vgg16.VGG16), ("VGG19", vgg19, vgg19.VGG19)]:
    model = cls(weights='imagenet')
    processed = module.preprocess_input(x.copy())
    model.predict(processed, verbose=0)          # warm-up run
    start = time.time()
    predictions = model.predict(processed, verbose=0)
    elapsed = time.time() - start
    print("\n%s (%s params, %.3f s per prediction)" % (name, format(model.count_params(), ","), elapsed))
    for _, label, prob in module.decode_predictions(predictions)[0]:
        print("  %-28s %.4f" % (label, prob))
