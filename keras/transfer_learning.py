import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os
import random

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

#download data 
#wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
#tar -xvzf 101_ObjectCategories.tar.gz

DATA_PATH = '/data/vision/fisher/data1/vsmolyakov/transfer_learning/'

exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces']
train_split, val_split = 0.7, 0.15

def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


categories = [x[0] for x in os.walk(DATA_PATH + '/101_ObjectCategories/') if x[0]][1:]
categories = [c for c in categories if c not in [os.path.join(DATA_PATH + '/101_ObjectCategories/', e) for e in exclude]]

#load all images
data = []
for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(category) 
              for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
    for img_path in images:
        img, x = get_image(img_path)
        data.append({'x': np.array(x[0]), 'y':c})

num_classes = len(categories)
random.shuffle(data)

#create train / val / test split
idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

#separate data and labels
x_train, y_train = np.array([t['x'] for t in train]), [t['y'] for t in train]
x_val, y_val = np.array([t['x'] for t in val]), [t['y'] for t in val]
x_test, y_test = np.array([t['x'] for t in test]), [t['y'] for t in test]

#normalize data
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print "finished loading %d images from %d categories" %(len(data), num_classes)
print "train / val / test split: %d, %d, %d" %(len(x_train), len(x_val), len(x_test))
print "training data shape: ", x_train.shape
print "training label shape: ", y_train.shape


#instantiate pre-trained architecture
resnet = ResNet50(weights='imagenet', include_top=True)

inp = resnet.input
top_layer = Dense(num_classes, activation='softmax')
out = top_layer(resnet.layers[-2].output)
resnet_new = Model(inp, out)

#freeze all layers except for the last one
for l, layer in enumerate(resnet_new.layers[:-1]):
    layer.trainable = False

#ensure the last layer is trainable
for l, layer in enumerate(resnet_new.layers[-1:]):
    layer.trainable = True

resnet_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
resnet_new.summary()

#create callbacks
file_name = DATA_PATH + 'resnet-new-weights-checkpoint.hdf5'
checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensor_board = TensorBoard(log_dir='./logs', write_graph=False, write_images=False)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=8, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=16, verbose=1)
callbacks_list = [checkpoint, tensor_board, reduce_lr, early_stopping]

#train the model
hist_resnet_new = resnet_new.fit(x_train, y_train, batch_size=32, epochs=128, verbose=2, validation_data=(x_val, y_val), callbacks=callbacks_list)

#save the model and weights
resnet_new.save(DATA_PATH + 'resnet_new_final_model.h5', overwrite=True)
resnet_new.save_weights(DATA_PATH + 'resnet_new_final_weights.h5', overwrite=True)

#evaluate the model on test data
test_loss, test_acc = resnet_new.evaluate(x_test, y_test, verbose=1)
print "Test loss: ", test_loss
print "Test accuracy: ", test_acc


#generate plots
plt.figure()
plt.plot(hist_resnet_new.history['val_loss'], label='ResNet50 new')
plt.title('ResNet50 Transfer Learning')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.savefig('./figures/resnet50_new_val_loss.png')

plt.figure()
plt.plot(hist_resnet_new.history['val_acc'], label='ResNet50 new')
plt.title('ResNet50 Transfer Learning')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.savefig('./figures/resnet50_new_val_acc.png')


