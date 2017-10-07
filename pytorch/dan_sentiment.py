import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable

from tqdm import tqdm
from time import time
from nltk.tokenize import RegexpTokenizer

SAVE_PATH = '/data/vision/fisher/data1/vsmolyakov/pytorch/sentiment_clf.pt'
EMBEDDINGS_FILE = '../word_vectors.txt'
MAX_SENT_LEN = 36  #max number of words per sentence
NUM_CLASSES = 2    #binary sentiment label

torch.manual_seed(0)

def get_embeddings():
    lines = []
    with open(EMBEDDINGS_FILE, 'r') as f:
        lines = f.readlines()
        f.close()
    
    embedding_tensor = []
    word_to_idx = {}
    
    for idx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb]
        if idx == 0: #reserved
            embedding_tensor.append(np.zeros(len(vector)))
        embedding_tensor.append(vector)
        word_to_idx[word] = idx+1
    #end for
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)    
    return embedding_tensor, word_to_idx
        
def get_tensor_idx(text, word_to_idx, max_len):
    null_idx = 0  #idx if word is not in the embeddings dictionary
    text_idx = [word_to_idx[x] if x in word_to_idx else null_idx for x in text][:max_len]
    if len(text_idx) < max_len:
        text_idx.extend([null_idx for _ in range(max_len - len(text_idx))])    
    x = torch.LongTensor(text_idx)  #64-bit integer
    return x
        

#load data
print "loading data..."
tic = time()
train_file = "../data/stsa.binary.train"
train_df = pd.DataFrame(columns=('sentiment', 'review'))
text_file = open(train_file, 'r')
lines = text_file.readlines()
for idx, review in enumerate(lines):
    train_df.loc[idx] = [review.split(' ')[0], ' '.join(review.split('\n')[0].split(' ')[1:])]    
text_file.close()
train_df.sentiment = train_df.sentiment.astype(np.int32)

val_file = "../data/stsa.binary.dev"
val_df = pd.DataFrame(columns=('sentiment', 'review'))
text_file = open(val_file, 'r')
lines = text_file.readlines()
for idx, review in enumerate(lines):
    val_df.loc[idx] = [review.split(' ')[0], ' '.join(review.split('\n')[0].split(' ')[1:])]    
text_file.close()
val_df.sentiment = val_df.sentiment.astype(np.int32)

test_file = "../data/stsa.binary.test"
test_df = pd.DataFrame(columns=('sentiment', 'review'))
text_file = open(test_file, 'r')
lines = text_file.readlines()
for idx, review in enumerate(lines):
    test_df.loc[idx] = [review.split(' ')[0], ' '.join(review.split('\n')[0].split(' ')[1:])]    
text_file.close()
test_df.sentiment = test_df.sentiment.astype(np.int32)
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "loading embeddings..."
tic = time()
embeddings, word_to_idx = get_embeddings()
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "generating training, validation, test datasets..."
tic = time()
tokenizer = RegexpTokenizer(r'\w+')
train_sent_len = []

train_data = []
for idx in range(train_df.shape[0]):
    label = train_df.iloc[idx,:]['sentiment']
    label_onehot = np.eye(NUM_CLASSES)[label]
    text = train_df.iloc[idx,:]['review']
    tokens = tokenizer.tokenize(text)[:MAX_SENT_LEN]
    train_sent_len.append(len(tokens))
    x = get_tensor_idx(tokens, word_to_idx, MAX_SENT_LEN)
    sample = {'x': x, 'y': torch.from_numpy(np.array([label], dtype=np.int64))}
    train_data.append(sample)
#end for

val_data = []
for idx in range(val_df.shape[0]):
    label = val_df.iloc[idx,:]['sentiment']
    label_onehot = np.eye(NUM_CLASSES)[label]
    text = val_df.iloc[idx,:]['review']
    tokens = tokenizer.tokenize(text)[:MAX_SENT_LEN]
    x = get_tensor_idx(tokens, word_to_idx, MAX_SENT_LEN)
    sample = {'x': x, 'y': torch.from_numpy(np.array([label], dtype=np.int64))}
    val_data.append(sample)
#end for

test_data = []
for idx in range(test_df.shape[0]):
    label = test_df.iloc[idx,:]['sentiment']
    label_onehot = np.eye(NUM_CLASSES)[label]
    text = test_df.iloc[idx,:]['review']
    tokens = tokenizer.tokenize(text)[:MAX_SENT_LEN]
    x = get_tensor_idx(tokens, word_to_idx, MAX_SENT_LEN)
    sample = {'x': x, 'y': torch.from_numpy(np.array([label], dtype=np.int64))}
    test_data.append(sample)
#end for
print "avg train sentence length: %.2f (+/- %.2f) words" %(np.mean(train_sent_len), np.std(train_sent_len))
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

#training parameters
num_epochs = 50
batch_size = 100

#model parameters
hidden_size = 300/2   #[300/4, 300/2, 300, 600]
weight_decay = 1e-3   #[1e-5, 1e-3, 1e1]
learning_rate = 1e-3  #[1e-5, 1e-3, 1e-1, 1e1]

#DNN architecture
class DAN(nn.Module):
    
    def __init__(self, embeddings, hidden_size, num_classes):
        super(DAN, self).__init__()
        
        vocab_size, embed_dim = embeddings.shape
        self.hidden_dim = hidden_size
        self.num_classes = num_classes

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        
        self.hidden = nn.Linear(embed_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.num_classes)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, x_idx):
        all_x = self.embedding_layer(x_idx)
        avg_x = torch.mean(all_x, dim=1)
        h1 = F.tanh(self.hidden(avg_x))
        out = self.out(h1)
        out = self.softmax(out)
        
        return out
        
use_gpu = torch.cuda.is_available()


model = DAN(embeddings, hidden_size, NUM_CLASSES)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
    
print model

#define loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

training_loss = []
validation_loss = []

print "training..."
for epoch in range(num_epochs):
    
    running_train_loss = 0.0
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4, 
        drop_last = True)
        
    model.train()
        
    for batch in tqdm(train_data_loader):
        
        x = Variable(batch['x'])
        y_temp = batch['y'].numpy().ravel()
        y_batch = torch.from_numpy(y_temp)
        y = Variable(y_batch)    

        if use_gpu:
            x, y = x.cuda(), y.cuda()
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
                
        running_train_loss += loss.cpu().data[0]        
        
    #end for
    training_loss.append(running_train_loss)
    print "epoch: %4d, training loss: %.4f" %(epoch+1, running_train_loss)
    
    #validation loss
    running_val_loss = 0.0
    
    val_data_loader = torch.utils.data.DataLoader(
        val_data, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4, 
        drop_last = True)
        
    model.eval()
    
    for batch in tqdm(val_data_loader):
        
        x = Variable(batch['x'])
        y_temp = batch['y'].numpy().ravel()
        y_batch = torch.from_numpy(y_temp)
        y = Variable(y_batch)    

        if use_gpu:
            x, y = x.cuda(), y.cuda()
            
        out = model(x)
        loss = criterion(out, y)
                
        running_val_loss += loss.cpu().data[0]        
        
    #end for
    validation_loss.append(running_val_loss)
    print "epoch: %4d, validation loss: %.4f" %(epoch+1, running_val_loss)
    
    torch.save(model, SAVE_PATH)
#end for


#test loss
predicted_total = 0
predicted_correct = 0
running_test_loss = 0.0
    
test_data_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4, 
    drop_last = True)
        
model.eval()
    
for batch in tqdm(test_data_loader):
        
    x = Variable(batch['x'])
    y_temp = batch['y'].numpy().ravel()
    y_batch = torch.from_numpy(y_temp)
    y = Variable(y_batch)    
       
    if use_gpu:
        x, y = x.cuda(), y.cuda()
            
    out = model(x)
    loss = criterion(out, y)                
    running_test_loss += loss.cpu().data[0]        
    
    _, predicted = torch.max(out.data, 1)
    predicted_total += y_temp.shape[0]
    if use_gpu:
        predicted_correct += (predicted.cpu().numpy() == y_temp).sum()
    else:
        predicted_correct += (predicted.numpy() == y_temp).sum()

print "test loss: %.4f" %(running_test_loss)
print "test accuracy: %.4f" %(predicted_correct / np.float(predicted_total))

#generate plots
plt.figure()
plt.plot(training_loss, label='Adam')
plt.title("Sentiment Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('./training_loss.png')

plt.figure()
plt.plot(validation_loss, label='Adam')
plt.title("Sentiment Model Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig('./validation_loss.png')

