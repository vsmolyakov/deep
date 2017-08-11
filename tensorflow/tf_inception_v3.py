#For more info see:
#https://github.com/tensorflow/models/blob/master/slim/slim_walkthrough.ipynb

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

import imagenet
import inception_preprocessing
from tensorflow.contrib.slim.nets import inception

import os
import json
import urllib2 as urllib
import tarfile

import PIL
import seaborn as sns
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

image_size = inception.inception_v3.default_image_size

with tf.Graph().as_default():
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    image_string = urllib.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_image = tf.expand_dims(processed_image, 0)

    #create the model using the default arg scope
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_image, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn('./tmp/inception_v3/inception_v3.ckpt', slim.get_model_variables('InceptionV3'))
 
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0,0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print "Prob %0.2f -> [%s]" %(probabilities[index]*100, names[index])

