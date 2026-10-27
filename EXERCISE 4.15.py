# Exercise 4.15
# Modify Example 4.17 (Part 4): visualize the second convolutional layer
# by changing n = 1 to n = 2, and comment on the results.
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt

model = VGG19()

n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Layer: ", model.layers[n].name)
print("Input channels: ", s[2])
print("Filter size: ", s[0], s[1])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Each filter in layer 2 has 64 input channels (one per filter of layer 1),
# so only the first 8 channels are drawn to keep the plot readable.
n_channels = 8
n_filters, ix = 4, 1
plt.figure(figsize=(12, 6))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(n_channels):
        ax = plt.subplot(n_filters, n_channels, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
plt.suptitle("First 4 filters of %s (first %d of %d channels)"
             % (model.layers[n].name, n_channels, s[2]))
plt.show()

# Comment:
# Layer 1 filters have 3 channels (R, G, B) because they look at the image
# itself, so they show simple colour/edge detectors. Layer 2 filters have
# 64 channels because they look at the 64 feature maps of layer 1. They
# still are 3x3, but each one combines the edges found by layer 1, so the
# individual channels look less meaningful on their own.
