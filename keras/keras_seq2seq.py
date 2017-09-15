import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Activation, Dense, RepeatVector, Input, merge
from keras.optimizers import Adam
from keras.models import load_model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

import io
import json
import math
from time import time
from unidecode import unidecode
from nltk.translate import bleu_score

DATA_PATH = "/data/vision/fisher/data1/vsmolyakov/seq2seq/"

def get_texts(source_texts, target_texts, max_len, max_examples):

    sources, targets = [], []
    for i, source in enumerate(source_texts):
        if len(source.split(' ')) <= max_len:
            target = target_texts[i]
            if len(target.split(' ')) <= max_len:
                sources.append(source)
                targets.append(target)
    return sources[:max_examples], targets[:max_examples]

def seq2seq_model(params):

    max_input_length  = params['max_input_length']
    max_output_length = params['max_output_length'] 
    source_vocab_size = params['source_vocab_size'] 
    target_vocab_size = params['target_vocab_size']
    embedding_dim     = params['embedding_dim']
    hidden_dim        = params['hidden_dim']

    input = Input(shape=(max_input_length,), dtype='int32')
    embed = Embedding(source_vocab_size, embedding_dim, input_length=max_input_length)(input)

    #bidirectional RNN
    forwards = LSTM(hidden_dim, return_sequences=False)(embed)
    backwards = LSTM(hidden_dim, return_sequences=False, go_backwards=True)(embed)
    encoder = merge([forwards, backwards], mode='concat', concat_axis=-1)

    #repeate encoder output for each output from the decoder
    encoder = RepeatVector(max_output_length)(encoder)

    decoder = LSTM(hidden_dim, return_sequences=True)(encoder)

    #apply dense layer to each timestamp
    decoder = TimeDistributed(Dense(target_vocab_size))(decoder)

    predictions = Activation("softmax")(decoder)

    return Model(input=input, output=predictions)

def decode_outputs(predictions, target_reverse_word_index):
    outputs = []
    for probs in predictions:
        preds = probs.argmax(axis=-1)
        tokens = []
        for idx in preds:
            tokens.append(target_reverse_word_index.get(idx))
        outputs.append(" ".join([t for t in tokens if t is not None]))
    return outputs

def build_seq_vecs(sequences):
    return np.array(sequences)

def build_target_vecs(tgt_texts, n_examples, target_tokenizer, max_output_length, target_vocab_size):
    y = np.zeros((n_examples, max_output_length, target_vocab_size), dtype=np.bool)
    for i, sent in enumerate(pad_sequences(target_tokenizer.texts_to_sequences(tgt_texts), maxlen=max_output_length)):
        word_idxs = np.arange(max_output_length)
        y[i][[word_idxs, sent]] = True
    return y

def generate_batches(params):

    batch_size = params['batch_size']
    n_examples = params['n_examples'] 
    src_texts = params['src_texts']
    tgt_texts = params['tgt_texts']
    source_tokenizer = params['source_tokenizer'] 
    target_tokenizer = params['target_tokenizer']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']
    target_vocab_size = params['target_vocab_size']

    gen_count = 0
    n_batches = np.int(math.floor(n_examples/batch_size))
    while True:
        sequences = pad_sequences(source_tokenizer.texts_to_sequences(src_texts), maxlen=max_input_length)
        X = build_seq_vecs(sequences)
        y = build_target_vecs(tgt_texts, n_examples, target_tokenizer, max_output_length, target_vocab_size)

        idx = np.random.permutation(len(sequences))
        X = X[idx]
        y = y[idx]

        for i in range(n_batches):
            gen_count += 1
            #print "yielding count: " + str(gen_count)
            start = batch_size * i
            end = start + batch_size
            yield X[start:end], y[start:end]
    #end while

def translate(model, sentences, params): 
    source_tokenizer = params['source_tokenizer']
    max_input_length = params['max_input_length']
    target_reverse_word_index = params['target_reverse_word_index']

    seqs = pad_sequences(source_tokenizer.texts_to_sequences(sentences), maxlen=max_input_length)
    input = build_seq_vecs(seqs)
    preds = model.predict(input, verbose=0)
    return decode_outputs(preds, target_reverse_word_index)

#load data (english to german)
print "loading and processing data..."
data = json.load(open('./data/en_de_corpus.json', 'r'))
word_len_source = [len(item.split(' ')) for item in data['en']]
word_len_target = [len(item.split(' ')) for item in data['de']]
word_len_max = int(max(np.mean(word_len_source), np.mean(word_len_target)) + 
                   max(np.std(word_len_source), np.std(word_len_target)))

#corpus parameters (english to german)
max_len = word_len_max          #max number of words
max_examples = len(data['en'])  #max number of examples
max_vocab_size = 10000          #max vocabulary size

src_all, tgt_all = get_texts(data['en'], data['de'], max_len, max_examples)

test_idx = int(0.999*len(src_all))
src_texts, src_test = src_all[:test_idx], src_all[test_idx:]
tgt_texts, tgt_test = tgt_all[:test_idx], tgt_all[test_idx:]

"""
#load data (english to russian)
print "loading and processing data..."

tic = time()
text_file = io.open(DATA_PATH + '/data/corpus.en_ru.1m.en', mode='r', encoding='UTF-8')
lines = text_file.readlines()
lines_io = io.StringIO(u" ".join(lines))
data_en = pd.read_csv(lines_io, delimiter='\n', header=None)
data_en = data_en.apply(lambda x: unidecode(x[0].decode('utf-8')), axis=1)
text_file.close()
toc = time()
print "english corpus done in %4.2f sec" %(toc-tic)

tic = time()
text_file = io.open(DATA_PATH + '/data/corpus.en_ru.1m.ru', mode='r', encoding='UTF-8')
lines = text_file.readlines()
lines_io = io.StringIO(u" ".join(lines))
data_ru = pd.read_csv(lines_io, delimiter='\n', header=None)
data_ru = data_ru.apply(lambda x: unidecode(x[0].decode('utf-8')), axis=1)
text_file.close()
toc = time()
print "russian corpus done in %4.2f sec" %(toc-tic)

data_en = data_en.values.tolist()
data_ru = data_ru.values.tolist()

word_len_source = [len(item.split(' ')) for item in data_en]
word_len_target = [len(item.split(' ')) for item in data_ru]
word_len_max = int(max(np.mean(word_len_source), np.mean(word_len_target)) + 
                   max(np.std(word_len_source), np.std(word_len_target)))

#corpus parameters (english to russian)
max_len = word_len_max / 4      #max number of words
max_examples = len(data_en)     #max number of examples
max_vocab_size = 10000          #max vocabulary size

src_all, tgt_all = get_texts(data_en, data_ru, max_len, max_examples)

test_idx = int(0.999*len(src_all))
src_texts, src_test = src_all[:test_idx], src_all[test_idx:]
tgt_texts, tgt_test = tgt_all[:test_idx], tgt_all[test_idx:]
"""

n_examples = len(src_texts)
print "number of training examples: ", n_examples
print "max word length: ", max_len 
print "max vocab size: ", max_vocab_size 

#model parameters
hidden_dim = 128
embedding_dim = 128 

#training parameters
n_epochs = 256 
batch_size = 32 
n_batches = np.int(math.floor(n_examples/batch_size))

start_token = '^'
end_token = '$'
src_texts = [' '.join([start_token, unidecode(text), end_token]) for text in src_texts]
tgt_texts = [' '.join([start_token, unidecode(text), end_token]) for text in tgt_texts]

src_test  = [' '.join([start_token, unidecode(text), end_token]) for text in src_test]
tgt_test  = [' '.join([start_token, unidecode(text), end_token]) for text in tgt_test]

print "tokenizing..."
filter_chars = '!"#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n\`""-'.replace(start_token, '').replace(end_token, '')
source_tokenizer = Tokenizer(max_vocab_size, filters=filter_chars)
source_tokenizer.fit_on_texts(src_texts)
target_tokenizer = Tokenizer(max_vocab_size, filters=filter_chars)
target_tokenizer.fit_on_texts(tgt_texts)

source_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1
print "source vocab size: ", source_vocab_size
print "target vocab size: ", target_vocab_size

max_input_length = max(len(seq) for seq in source_tokenizer.texts_to_sequences_generator(src_texts))
max_output_length = max(len(seq) for seq in source_tokenizer.texts_to_sequences_generator(tgt_texts))
target_reverse_word_index = {v:k for k, v in target_tokenizer.word_index.items()}

seq2seq_params = {
    'max_input_length':  max_input_length, 
    'max_output_length': max_output_length,
    'source_vocab_size': source_vocab_size,
    'target_vocab_size': target_vocab_size,
    'embedding_dim':     embedding_dim,
    'hidden_dim':        hidden_dim
}

generator_params = {
    'batch_size':        batch_size,
    'n_examples':        n_examples,
    'src_texts':         src_texts,
    'tgt_texts':         tgt_texts,
    'source_tokenizer':  source_tokenizer,
    'target_tokenizer':  target_tokenizer,
    'max_input_length':  max_input_length,
    'max_output_length': max_output_length,
    'target_vocab_size': target_vocab_size
}

translate_params = {
    'source_tokenizer':  source_tokenizer,
    'max_input_length':  max_input_length,
    'target_reverse_word_index': target_reverse_word_index
}

print "compiling seq2seq model..."
model = seq2seq_model(seq2seq_params)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

#define checkpoints 
file_name = DATA_PATH + 'weights_improvement_en2de.hdf5'
checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensor_board = TensorBoard(log_dir='./logs', write_graph=False, write_images=False)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=8, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=16, verbose=1)
callbacks_list = [checkpoint, tensor_board, reduce_lr, early_stopping]

print "training seq2seq model..."
hist = model.fit_generator(generator=generate_batches(generator_params), steps_per_epoch=n_batches, epochs=n_epochs, verbose=2, callbacks=callbacks_list)

model.save(DATA_PATH + "seq2seq_model_en2de.h5", overwrite=True)
model.save_weights(DATA_PATH + "seq2seq_weights_en2de.h5", overwrite=True)

#load saved model
#model = load_model(DATA_PATH + '/trained_models/seq2seq_model_en2de.h5')

#generate output
print src_test[0]
print tgt_test[0]
print translate(model, [src_test[0]], translate_params)

#compute corpus bleu score
hypotheses = []
references = []
for idx, sent in enumerate(src_test):
    hyp = translate(model, [sent], translate_params)
    ref = [tgt_test[idx].lower()]
    hypotheses.append(hyp)
    references.append(ref)

corp_bleu_score = bleu_score.corpus_bleu(references, hypotheses)
print "test corpus bleu score: ", corp_bleu_score

#generate plots
plt.figure()
plt.plot(hist.history['loss'], label='Adam')
plt.title('seq2seq training')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig('./figures/seq2seq_en2de_training_loss.png')

plt.figure()
plt.plot(hist.history['acc'], label='Adam')
plt.title('seq2seq training')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.savefig('./figures/seq2seq_en2de_training_acc.png')

