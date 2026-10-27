# Example 4.17 CNN Visualize Filters.ipynb
#Modified from:
#https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
# The book's six notebook cells (Part 1-6) are combined into one script.
# Part 6 needs an image named Elephant.jpg in the same folder as this script.
import os
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

# Part 1: load and summarize the VGG19 model ============================
model = VGG19()
model.summary()

# Part 2: list all layers with their numbers ============================
n = 0
for layer in model.layers:
    print(n, layer.name)
    n += 1

# Part 3: filter and bias shapes of the convolutional layers ============
n = 0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n, layer.name, filters.shape, biases.shape)
    n += 1

# Part 4: plot the filters of a convolutional layer =====================
# filters has shape (rows, cols, channels, number of filters). The book
# prints s[0] as the colour channels and plots f[j, :, :]; the channels
# are actually s[2] and each one is f[:, :, j], fixed here.
# retrieve weights from the first hidden layer
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[2])
print("Filter size: ", s[0], s[1])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
#n_filters, ix = s[3], 1
n_filters, ix = 4, 1
plt.figure(figsize=(10, 10))
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(s[2]):
        ax = plt.subplot(n_filters, s[2], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
# show the figure
plt.show()

# Part 5: function to draw feature maps in a grid =======================
def plot_feature_maps(feature_maps):
    # plot all feature maps
    col = 8
    row = int(feature_maps.shape[3] / col)
    ix = 1
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()

# Part 6: feature maps (output) of a hidden layer for an image ==========
# load the model
model = VGG19()
#Select the hidden layer to visualize
n = 1
# redefine model to output right after the hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
model.summary()

# load the image with the required shape
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
img = load_img(img_path, target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
print("Feature maps: ", feature_maps.shape)
#plot all the feature maps
plot_feature_maps(feature_maps)
