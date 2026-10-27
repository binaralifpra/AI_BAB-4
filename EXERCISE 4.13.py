# Exercise 4.13
# Modify Example 4.15, try the different models and compare their performances.
# image-classifiers does not run on Keras 3, so the same five models are
# loaded from keras.applications. Every model classifies the same webcam
# frame and the results are printed as a table: parameters, prediction
# time and top prediction.
import time
import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg16, resnet50, mobilenet_v2, densenet, inception_v3

# name: (constructor, preprocess module, input size)
MODELS = {
    'vgg16':       (vgg16.VGG16, vgg16, 224),
    'resnet50':    (resnet50.ResNet50, resnet50, 224),
    'mobilenetv2': (mobilenet_v2.MobileNetV2, mobilenet_v2, 224),
    'densenet201': (densenet.DenseNet201, densenet, 224),
    'inceptionv3': (inception_v3.InceptionV3, inception_v3, 299),
}

# grab one frame from the webcam so every model sees the same image
camera = cv2.VideoCapture(0)
ok, cam_frame = camera.read()
camera.release()
if not ok:
    raise SystemExit("Could not read from the webcam")
rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)

print("%-12s %-14s %-10s %-26s %s" % ("Model", "Params", "Time (s)", "Prediction", "Prob"))
for name, (constructor, net, sz) in MODELS.items():
    model = constructor(weights='imagenet')
    image = np.expand_dims(cv2.resize(rgb, (sz, sz)).astype('float32'), 0)
    image = net.preprocess_input(image)
    model.predict(image, verbose=0)               # warm-up run
    start = time.time()
    preds = model.predict(image, verbose=0)
    elapsed = time.time() - start
    _, label, prob = decode_predictions(preds)[0][0]
    print("%-12s %-14s %-10.3f %-26s %.3f" % (name, format(model.count_params(), ","),
                                             elapsed, label, prob))
