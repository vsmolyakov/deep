import numpy as np
import pandas as pd

import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from keras.applications import VGG16 
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3

from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator 

MODELS = {
    'vgg16': VGG16,
    'vgg19': VGG19,
    'inception': InceptionV3,
    'resnet': ResNet50
}

arch = 'vgg16'
image_path = './figures/dogs.jpg'

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if arch == 'inception':
    input_shape = (299, 299)
    preprocess = preprocess_input

Network = MODELS[arch]
model = Network(weights="imagenet")

print "\nloading image..."
image = load_img(image_path, target_size=input_shape)
image = img_to_array(image)
image = np.expand_dims(image, axis=0) # 1 x input_shape
image = preprocess(image)

print "\nclassifying image with %s ..." %arch
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds, top=5)
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print "%s %s: %.2f" %(i+1, label, prob * 100)

#class activation map (grad-cam)
top_pred_idx = np.argmax(preds[0])

top_pred_output = model.output[:, top_pred_idx]

last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(top_pred_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([image])

num_channels = conv_layer_output_value.shape[-1]

#multiply each channel feature map by mean gradient of the class wrt to the channel
for i in range(num_channels):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1) 
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
#plt.matshow(heatmap)

#overlay heatmap and the original image
example = cv2.imread(image_path)
heatmap = cv2.resize(heatmap, (example.shape[1], example.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

example_cam = heatmap * 0.7 + example
cv2.imwrite('./figures/dogs_cam.png', example_cam)


