# Exercise 4.17
# Modify Example 4.17 (Part 6): visualize the second, third and fourth
# convolutional layers by changing n = 1 to n = 2 and so on, and comment.
# Needs an image named Elephant.jpg in the same folder as this script.
import os
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

def plot_feature_maps(feature_maps, title):
    # plot the first 64 feature maps in an 8 x 8 grid
    col = 8
    row = min(int(feature_maps.shape[3] / col), 8)
    ix = 1
    plt.figure(figsize=(12, 12))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.suptitle(title)
    plt.show()

vgg = VGG19()

img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
img = img_to_array(load_img(img_path, target_size=(224, 224)))
img = preprocess_input(expand_dims(img, axis=0))

# layer numbers of the 2nd, 3rd and 4th convolutional layers
# (2 = block1_conv2, 4 = block2_conv1, 5 = block2_conv2)
for n in [2, 4, 5]:
    model = Model(inputs=vgg.inputs, outputs=vgg.layers[n].output)
    feature_maps = model.predict(img, verbose=0)
    name = vgg.layers[n].name
    print("Layer %d (%s) feature maps: %s" % (n, name, feature_maps.shape))
    plot_feature_maps(feature_maps, "Layer %d: %s %s" % (n, name, feature_maps.shape))

# Comment:
# Layer 1 and 2 feature maps (224x224) still look like the elephant, with
# edges and colours highlighted. After the first pooling, layers 4 and 5
# are 112x112 with 128 maps; they are blurrier and each map responds to
# more specific patterns (texture, outlines). The deeper the layer, the
# smaller and more abstract the feature maps become.
