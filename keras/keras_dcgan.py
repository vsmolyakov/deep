import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Model 
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, LeakyReLU 

from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing import image

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler 
from keras.callbacks import EarlyStopping

import os, math
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sns.set_style("whitegrid")

FIGURES_PATH = "./figures/"

#load data
print "loading data..."
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 7] #select horse images
x_train = x_train.reshape((x_train.shape[0],) + (32, 32, 3)).astype('float32')/255.0

#training params
iterations = 10000
batch_size = 20

#model params
latent_dim = 32
height, width, channels = 32, 32, 3

#generator architecture
generator_input = Input(shape=(latent_dim, ))

x = Dense(128 * 16 * 16)(generator_input)
x = LeakyReLU()(x)
x = Reshape((16, 16, 128))(x)

x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(channels, 7, activation='tanh', padding='same')(x) #NOTE: tanh
generator = Model(generator_input, x)
generator.summary()

#discriminator architecture
discriminator_input = Input(shape=(height, width, channels))
x = Conv2D(128, 3)(discriminator_input)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)
x = Flatten()(x)

x = Dropout(0.4)(x) #important
x = Dense(1, activation='sigmoid')(x)
discriminator = Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

#GAN architecture
discriminator.trainable = False  #applies to GAN only (since discriminator is compiled)
gan_input = Input(shape=(latent_dim, ))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

gan_optimizer = optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

#GAN training
start = 0
for step in tqdm(range(iterations)):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape) #important

    d_loss = discriminator.train_on_batch(combined_images, labels)
    
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1)) #says all images are real

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    if step % 1000 == 0:
        #gan.save_weights('gan.h5')
        print "step ", step
        print "discriminator loss: ", d_loss
        print "adversarial loss: ", a_loss

        img = image.array_to_img(generated_images[0] * 255.0, scale=False)
        img.save(FIGURES_PATH + '/generated_horse_iter_' + str(step) + '.png')

        img = image.array_to_img(real_images[0] * 255.0, scale=False)
        img.save(FIGURES_PATH + '/real_horse_iter_' + str(step) + '.png')
    #end if
#end for





















