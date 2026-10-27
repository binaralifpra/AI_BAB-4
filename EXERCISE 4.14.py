# Exercise 4.14
# Modify Example 4.15 to use an if-else statement to select the model.
# image-classifiers does not run on Keras 3, so the models come from
# keras.applications. Press ESC to quit.
import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg16, resnet50, mobilenet_v2, densenet, inception_v3

print("Models: vgg16, resnet50, mobilenetv2, densenet201, inceptionv3")
choice = input("Select a model: ").strip().lower()

sz = 224
if choice == 'vgg16':
    model, preprocess_input = vgg16.VGG16(weights='imagenet'), vgg16.preprocess_input
elif choice == 'resnet50':
    model, preprocess_input = resnet50.ResNet50(weights='imagenet'), resnet50.preprocess_input
elif choice == 'mobilenetv2':
    model, preprocess_input = mobilenet_v2.MobileNetV2(weights='imagenet'), mobilenet_v2.preprocess_input
elif choice == 'densenet201':
    model, preprocess_input = densenet.DenseNet201(weights='imagenet'), densenet.preprocess_input
elif choice == 'inceptionv3':
    model, preprocess_input = inception_v3.InceptionV3(weights='imagenet'), inception_v3.preprocess_input
    sz = 299
else:
    print("Unknown model, using vgg16")
    choice = 'vgg16'
    model, preprocess_input = vgg16.VGG16(weights='imagenet'), vgg16.preprocess_input
model.summary()

camera = cv2.VideoCapture(0)
while True:
    ret, cam_frame = camera.read()
    if not ret:
        break
    frame = cv2.cvtColor(cv2.resize(cam_frame, (sz, sz)), cv2.COLOR_BGR2RGB)
    image = np.expand_dims(frame.astype('float32'), 0)
    image = preprocess_input(image)
    preds = model.predict(image, verbose=0)
    label = decode_predictions(preds)
    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(choice, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
