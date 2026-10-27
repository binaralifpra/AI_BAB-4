#Exercise 4.11
# Modify the VGG16 webcam program (Example 4.13) to use VGG19, and compare
# performance. Frames per second (FPS) is shown on screen; run Example 4.13
# alongside to compare speed and the predicted labels. Press ESC to quit.
import time
import cv2
import numpy as np
from keras.utils import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg19

image_size = 224
model = vgg19.VGG19(weights='imagenet')
model.summary()

camera = cv2.VideoCapture(0)
while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    start = time.time()
    frame = cv2.cvtColor(cv2.resize(cam_frame, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    processed_image = vgg19.preprocess_input(image_batch.copy())

    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)
    fps = 1.0 / (time.time() - start)

    cv2.putText(cam_frame, "VGG19: {}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(cam_frame, "FPS: {:.1f}".format(fps),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
