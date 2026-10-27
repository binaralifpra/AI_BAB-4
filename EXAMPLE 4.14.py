# Example 4.14
# Classifies webcam frames with VGG16, ResNet50, MobileNet or Inception V3.
# Uncomment one model, its preprocess_input and the matching image_size
# (Inception V3 needs 299). Press ESC to quit.
import cv2
import numpy as np
from keras.utils import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

# init the models
#model = vgg16.VGG16(weights='imagenet')
model = resnet50.ResNet50(weights='imagenet')
#model = mobilenet.MobileNet(weights='imagenet')
#model = inception_v3.InceptionV3(weights='imagenet')
model.summary()

camera = cv2.VideoCapture(0)
image_size = 224
#image_size = 299
while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    # OpenCV gives BGR; Keras models expect RGB
    frame = cv2.cvtColor(cv2.resize(cam_frame, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    #processed_image = vgg16.preprocess_input(image_batch.copy())
    processed_image = resnet50.preprocess_input(image_batch.copy())
    #processed_image = mobilenet.preprocess_input(image_batch.copy())
    #processed_image = inception_v3.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)

    # format final image visualization to display the results of experiments
    cv2.putText(cam_frame, "{}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
