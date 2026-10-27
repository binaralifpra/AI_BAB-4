# Exercise 4.12
# Modify Example 4.14 and add DenseNet to classify webcam images, then
# compare the performances. Change MODEL_NAME to switch networks; the
# frames per second (FPS) on screen shows the speed of each model.
# Press ESC to quit.
import time
import cv2
import numpy as np
from keras.utils import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3,
    densenet
)

MODEL_NAME = 'densenet121'   # vgg16, resnet50, mobilenet, inceptionv3, densenet121

# name: (constructor, preprocess module, input size)
MODELS = {
    'vgg16':       (vgg16.VGG16, vgg16, 224),
    'resnet50':    (resnet50.ResNet50, resnet50, 224),
    'mobilenet':   (mobilenet.MobileNet, mobilenet, 224),
    'inceptionv3': (inception_v3.InceptionV3, inception_v3, 299),
    'densenet121': (densenet.DenseNet121, densenet, 224),
}

constructor, net, image_size = MODELS[MODEL_NAME]
model = constructor(weights='imagenet')
model.summary()
print("Total params:", format(model.count_params(), ","))

camera = cv2.VideoCapture(0)
while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    start = time.time()
    frame = cv2.cvtColor(cv2.resize(cam_frame, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    processed_image = net.preprocess_input(image_batch.copy())

    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)
    fps = 1.0 / (time.time() - start)

    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(MODEL_NAME, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(cam_frame, "FPS: {:.1f}".format(fps),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
