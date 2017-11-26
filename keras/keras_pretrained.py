import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import VGG16 
from keras.applications import VGG19

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

arch = 'inception'
image_path = './figures/zebra.jpg'

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if arch == 'inception':
    input_shape = (299, 299)
    preprocess = preprocess_input

Network = MODELS[arch]
model = Network(weights="imagenet")

#data augmentation
test_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

print "\nloading image..."
image = load_img(image_path, target_size=input_shape)
image = img_to_array(image)
#test_datagen.fit(np.expand_dims(image, axis=0)) #rank=4
#image = test_datagen.random_transform(image) #rank=3
image = np.expand_dims(image, axis=0) # 1 x input_shape
image = preprocess(image)

print "\nclassifying image with %s ..." %arch
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print "%s %s: %.2f" %(i+1, label, prob * 100)

