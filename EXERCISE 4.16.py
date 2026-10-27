# Exercise 4.16
# Modify Example 4.17 (Part 4): display all the filters by changing
# n_filters, ix = 4, 1 to n_filters, ix = s[3], 1, and comment on the results.
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt

model = VGG19()

n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[2])
print("Filter size: ", s[0], s[1])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot all 64 filters: one row per filter, one column per colour channel
n_filters, ix = s[3], 1
plt.figure(figsize=(4, 40))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[2]):
        ax = plt.subplot(n_filters, s[2], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
plt.show()

# Comment:
# All 64 filters of layer 1 give a tall 64 x 3 grid of small 3x3 images.
# Many filters have a light and a dark side (edge detectors in different
# directions), and some have very different values per channel, so they
# respond to a particular colour. Showing all 64 makes the plot crowded,
# which is why the book shows only the first 4.
