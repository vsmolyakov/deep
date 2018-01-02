# deep
deep learning

### Description

**Keras**

- [image_search](./keras/image_search.py): VGG-16 applied to Caltech101 dataset for nearest neighbor image retrieval
- [keras_lenet](./keras/keras_lenet.py): LeNet architecture for MNIST digit recognition
- [keras_vae](./keras/keras_vae.py): Variational Auto-Encoder (VAE) trained on MNIST digits
- [keras_dcgan](./keras/keras_dcgan_cifar10.py): Deep Convolutional GAN (DC-GAN) trained on CIFAR10 dataset
- [sentiment_kernel](./keras/sentiment_kernel.py): LSTM neural network for sentence sentiment prediction
- [lstm_language](./keras/keras_lstm_language.py): LSTM language model for generating text trained on Shakespeare
- [lstm_series](./keras/keras_lstm_series.py): LSTM time series prediction applied to S&P500 data
- [pretrained](./keras/keras_pretrained.py): pretrained VGG16, VGG19, ResNet50, InceptionV3 for image classification
- [transfer_learning](./keras/transfer_learning.py): ResNet50 fine tuned on Caltech101 dataset  
- [keras_seq2seq](./keras/keras_seq2seq.py): seq2seq model for machine translation with bidirectional RNN encoder  
- [grad_cam](./keras/keras_grad_cam.py): class activation map computed by weighing each channel by its average gradient  
- [style_transfer](./keras/keras_style_transfer.py): neural style transfer with VGG19 by minimizing content and style loss with L-BFGS  
- [keras_mdn](./keras/keras_mdn.py): Mixture Density Network (MDN) for learning parameters of a Gaussian Mixture Model

References:  
*https://keras.io/*  

**TensorFlow**

- [tf_classifier](./tensorflow/tf_classifier.py): DNN classifier for Iris dataset
- [tf_regressor](./tensorflow/tf_regressor.py): DNN regressor for estimating Boston housing prices
- [tf_mlp](./tensorflow/tf_mlp.py): Multi-Layer Perceptron with callbacks
- [tf_cnn_mnist](./tensorflow/tf_cnn_mnist.py): CNN for MNIST digit classification
- [tf_autoencoder](./tensorflow/tf_autoencoder.py): a two-layer encoder / decoder architecture applied to MNIST digits
- [tf_word2vec](./tensorflow/tf_word2vec.py): word2vec skip-gram model trained on the 20 newsgroups dataset
- [tf_wide_and_deep](./tensorflow/tf_wide_and_deep.py): wide and deep classification architecture trained on census income dataset
- [tf_gan_mlp](./tensorflow/tf_gan_mlp.py): generative adversarial network based on two MLPs using MNIST digits
- [tf_inception_v3](./tensorflow/tf_inception_v3.py): InceptionV3 architecture pre-trained on ILSVRC-2012-CLS image classification dataset
- [tf_optimizers](./tensorflow/tf_optimizers.py): a comparison of SGD, Momentum, RMSProp and Adam optimizers using a CNN trained on MNIST digits


References:  
*https://www.tensorflow.org/*

**PyTorch**
- [lenet5_cifar10](./pytorch/lenet5_cifar10.py): LeNet5 CNN architecture for CIFAR10 object classification   
- [dan_sentiment](./pytorch/dan_sentiment.py): sentiment classifier based on averaging of pretrained word embeddings  
- [lstm_qsim](./pytorch/lstm_qsim.py): LSTM encoder for question similarity trained on stack exchange dataset  
- [gradients](./pytorch/gradient_norm.py): computes gradient and weight norms for a simple MLP architecture with different optimizers  


References:
*http://pytorch.org/*


**Theano**
- [cnn_lenet](./theano/theano_cnn_lenet.py): LeNet CNN architecture for MNIST digit classification
- [mlp](./theano/mlp.py): Multi-Layer Perceptron
- [logistic_sgd](./theano/logistic_sgd.py): Logistic Regression
- [theano_linreg](./theano/theano_linreg.py): Linear Regression


References:
*http://deeplearning.net/software/theano/*


### Dependencies

Python 2.7  

