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
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from time import time
import cPickle as pickle
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(0)
torch.manual_seed(0)

#The askubuntu stack exchange dataset is available at: https://github.com/taolei87/askubuntu
DATA_PATH = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/askubuntu/'
SAVE_PATH = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/askubuntu/'
SAVE_NAME = './lstm_baseline.pt' 

EMBEDDINGS_FILE = DATA_PATH + '/vector/vectors_pruned.200.txt'
MAX_TITLE_LEN = 10
MAX_BODY_LEN = 100  #max number of words per sentence

tokenizer = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))

def compute_mrr(data_frame, score_name='bm25_score'):

    mrr_output = []
    for qidx in range(data_frame.shape[0]):
        retrieved_set = map(int, data_frame.loc[qidx, 'random_id'].split(' '))
        relevant_set = set(map(int, data_frame.loc[qidx, 'similar_id'].split(' ')))
        retrieved_scores = map(float, data_frame.loc[qidx, score_name].split(' '))

        #sort according to scores (higher score is better, i.e. ranked higher)        
        retrieved_set_sorted = [p for p, s in sorted(zip(retrieved_set, retrieved_scores),
                                key = lambda pair: pair[1], reverse=True)]

        rank = 1
        for item in retrieved_set_sorted:
            if item in relevant_set:
                break
            else:
                rank += 1
        #end for
        MRR = 1.0 / rank
        mrr_output.append(MRR)
    #end for
    return mrr_output

def precision_at_k(data_frame, K=5, score_name='bm25_score'):

    pr_output = []
    for qidx in range(data_frame.shape[0]):
        retrieved_set = map(int, data_frame.loc[qidx, 'random_id'].split(' '))
        relevant_set = set(map(int, data_frame.loc[qidx, 'similar_id'].split(' '))) 
        retrieved_scores = map(float, data_frame.loc[qidx, score_name].split(' '))

        #sort according to scores (higher score is better, i.e. ranked higher)        
        retrieved_set_sorted = [p for p, s in sorted(zip(retrieved_set, retrieved_scores),
                                key = lambda pair: pair[1], reverse=True)]

        count = 0
        for item in retrieved_set_sorted[:K]:
            if item in relevant_set:
                count += 1
        #end for
        precision_at_k = count / float(K)
        pr_output.append(precision_at_k)
    #end for
    return pr_output

def compute_map(data_frame, score_name='bm25_score'):

    map_output = []
    for qidx in range(data_frame.shape[0]):
        retrieved_set = map(int, data_frame.loc[qidx, 'random_id'].split(' '))
        relevant_set = set(map(int, data_frame.loc[qidx, 'similar_id'].split(' '))) 
        retrieved_scores = map(float, data_frame.loc[qidx, score_name].split(' '))

        #sort according to scores (higher score is better, i.e. ranked higher)        
        retrieved_set_sorted = [p for p, s in sorted(zip(retrieved_set, retrieved_scores),
                                key = lambda pair: pair[1], reverse=True)]

        AP = 0
        num_relevant = 0
        for ridx, item in enumerate(retrieved_set_sorted):
            if item in relevant_set:
                num_relevant += 1
                #compute precision at K=ridx+1
                count = 0
                for entry in retrieved_set_sorted[:ridx+1]:
                    if entry in relevant_set:
                        count += 1
                #end for
                AP += count / float(ridx+1)
            #end if
        #end for
        if (num_relevant > 0):
            AP = AP / float(num_relevant)
        else:
            AP = 0
        #end for
        map_output.append(AP)
    #end for
    return map_output


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
        
def generate_data(data_frame, train_text_df, word_to_idx, tokenizer, type='train'):

    dataset = []
    for idx in tqdm(range(100)):
    #for idx in tqdm(range(data_frame.shape[0])):
        query_id = data_frame.loc[idx, 'query_id']
        similar_id_list = map(int, data_frame.loc[idx, 'similar_id'].split(' '))
        random_id_list = map(int, data_frame.loc[idx, 'random_id'].split(' '))
    
        #query title and body tensor ids
        query_title = train_text_df[train_text_df['id'] == query_id].title.tolist() 
        query_body = train_text_df[train_text_df['id'] == query_id].body.tolist()
        query_title_tokens = tokenizer.tokenize(query_title[0])[:MAX_TITLE_LEN]
        query_body_tokens = tokenizer.tokenize(query_body[0])[:MAX_BODY_LEN]
        query_title_tensor_idx = get_tensor_idx(query_title_tokens, word_to_idx, MAX_TITLE_LEN) 
        query_body_tensor_idx = get_tensor_idx(query_body_tokens, word_to_idx, MAX_BODY_LEN)

        if (type != 'train'):
            similar_id_list = similar_id_list[:1] #keep only one element

        for similar_id in similar_id_list:
            sample = {}  #reset sample dictionary here
            sample['query_idx'] = query_id
            sample['query_title'] = query_title_tensor_idx
            sample['query_body'] = query_body_tensor_idx

            similar_title = train_text_df[train_text_df['id'] == similar_id].title.tolist() 
            similar_body = train_text_df[train_text_df['id'] == similar_id].body.tolist()
            similar_title_tokens = tokenizer.tokenize(similar_title[0])[:MAX_TITLE_LEN]
            similar_body_tokens = tokenizer.tokenize(similar_body[0])[:MAX_BODY_LEN]
            similar_title_tensor_idx = get_tensor_idx(similar_title_tokens, word_to_idx, MAX_TITLE_LEN) 
            similar_body_tensor_idx = get_tensor_idx(similar_body_tokens, word_to_idx, MAX_BODY_LEN)
            sample['similar_title'] = similar_title_tensor_idx
            sample['similar_body'] = similar_body_tensor_idx

            for ridx, random_id in enumerate(random_id_list):
                random_title_name = 'random_title_' + str(ridx)
                random_body_name = 'random_body_' + str(ridx)
        
                random_title = train_text_df[train_text_df['id'] == random_id].title.tolist() 
                random_body = train_text_df[train_text_df['id'] == random_id].body.tolist()
                
                if (len(random_title) > 0 and len(random_body) > 0):
                    random_title_tokens = tokenizer.tokenize(random_title[0])[:MAX_TITLE_LEN]
                    random_body_tokens = tokenizer.tokenize(random_body[0])[:MAX_BODY_LEN]
                    random_title_tensor_idx = get_tensor_idx(random_title_tokens, word_to_idx, MAX_TITLE_LEN) 
                    random_body_tensor_idx = get_tensor_idx(random_body_tokens, word_to_idx, MAX_BODY_LEN)
                    sample[random_title_name] = random_title_tensor_idx
                    sample[random_body_name] = random_body_tensor_idx
                else:
                    #generate a vector of all zeros (need 100 negative examples for each batch)
                    sample[random_title_name] = torch.zeros(MAX_TITLE_LEN).type(torch.LongTensor) 
                    sample[random_body_name] = torch.zeros(MAX_BODY_LEN).type(torch.LongTensor)
                #end if
            #end for
            dataset.append(sample)
        #end for
    #end for
    return dataset 


#load data
print "loading data..."
tic = time()
train_text_file = DATA_PATH + '/text_tokenized.txt'
train_text_df = pd.read_table(train_text_file, sep='\t', header=None)
train_text_df.columns = ['id', 'title', 'body']
train_text_df = train_text_df.dropna()
train_text_df['title'] = train_text_df['title'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
train_text_df['body'] = train_text_df['body'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
train_text_df['title_len'] = train_text_df['title'].apply(lambda words: len(tokenizer.tokenize(str(words))))
train_text_df['body_len'] = train_text_df['body'].apply(lambda words: len(tokenizer.tokenize(str(words))))

train_idx_file = DATA_PATH + '/train_random.txt' 
train_idx_df = pd.read_table(train_idx_file, sep='\t', header=None)
train_idx_df.columns = ['query_id', 'similar_id', 'random_id']
train_idx_df = train_idx_df.dropna()
train_idx_df = train_idx_df.reset_index()

dev_idx_file = DATA_PATH + '/dev.txt'
dev_idx_df = pd.read_table(dev_idx_file, sep='\t', header=None)
#dev_idx_df.columns = ['query_id', 'similar_id', 'retrieved_id', 'bm25_score']
dev_idx_df.columns = ['query_id', 'similar_id', 'random_id', 'bm25_score']
dev_idx_df = dev_idx_df.dropna()
dev_idx_df = dev_idx_df.reset_index()

test_idx_file = DATA_PATH + '/test.txt'
test_idx_df = pd.read_table(test_idx_file, sep='\t', header=None)
#test_idx_df.columns = ['query_id', 'similar_id', 'retrieved_id', 'bm25_score']
test_idx_df.columns = ['query_id', 'similar_id', 'random_id', 'bm25_score']
test_idx_df = test_idx_df.dropna()
test_idx_df = test_idx_df.reset_index()

toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "loading embeddings..."
tic = time()
embeddings, word_to_idx = get_embeddings()
print "vocab size (embeddings): ", len(word_to_idx)
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

#visualize data
f, (ax1, ax2) = plt.subplots(1, 2)
sns.distplot(train_text_df['title_len'], hist=True, kde=True, color='b', label='title len', ax=ax1)
sns.distplot(train_text_df[train_text_df['body_len'] < 256]['body_len'], hist=True, kde=True, color='r', label='body len', ax=ax2)
ax1.axvline(x=MAX_TITLE_LEN, color='k', linestyle='--', label='max len')
ax2.axvline(x=MAX_BODY_LEN, color='k', linestyle='--', label='max len')
ax1.set_title('title length histogram'); ax1.legend(loc=1); 
ax2.set_title('body length histogram'); ax2.legend(loc=1);
plt.savefig('../figures/question_len_hist.png')


print "generating training, validation, test datasets..."
tic = time()
train_data = generate_data(train_idx_df, train_text_df, word_to_idx, tokenizer, type='train')
val_data = generate_data(dev_idx_df, train_text_df, word_to_idx, tokenizer, type='dev')
test_data = generate_data(test_idx_df, train_text_df, word_to_idx, tokenizer, type='test')
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)


#training parameters
num_epochs = 16 
batch_size = 32 

#model parameters
embed_dim = embeddings.shape[1] #200
hidden_size = 128 
weight_decay = 1e-5 
learning_rate = 1e-3 

#RNN architecture
class RNN(nn.Module):
    
    def __init__(self, embed_dim, hidden_size, vocab_size, batch_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        #TODO: make sure loss is not computed for 0 padded values
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim) 
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        #[num_layers, batch_size, hidden_size] for (h_n, c_n)
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(2, self.batch_size, self.hidden_size)))

    def forward(self, x_idx):
        all_x = self.embedding_layer(x_idx)
        #[batch_size, seq_length (num_words), embed_dim]
        lstm_out, self.hidden = self.lstm(all_x.view(self.batch_size, x_idx.size(1), -1), self.hidden)
        h_avg_pool = torch.mean(lstm_out, dim=1)          #average pooling
        #h_n, c_n = self.hidden[0], self.hidden[1]        #last pooling
        #h_last_pool = torch.cat([h_n[0], h_n[1]], dim=1) #[batch_size, 2 x hidden_size] 

        return h_avg_pool 

use_gpu = torch.cuda.is_available()

model = RNN(embed_dim, hidden_size, len(word_to_idx), batch_size)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
    
print model

#define loss and optimizer
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
lstm_num_params = sum([np.prod(p.size()) for p in model_parameters])
print "number of trainable params: ", lstm_num_params

criterion = nn.MultiMarginLoss(p=1, margin=0.4, size_average=True)
optimizer = torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=4, gamma=0.5) #half learning rate every 4 epochs

learning_rate_schedule = [] 
training_loss, validation_loss, test_loss = [], [], []

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
    scheduler.step()
        
    for batch in tqdm(train_data_loader):
     
        query_title = Variable(batch['query_title'])
        query_body = Variable(batch['query_body'])
        similar_title = Variable(batch['similar_title'])
        similar_body = Variable(batch['similar_body'])

        random_title_list = []
        random_body_list = []
        for ridx in range(40): #number of random negative examples
            random_title_name = 'random_title_' + str(ridx)
            random_body_name = 'random_body_' + str(ridx)
            random_title_list.append(Variable(batch[random_title_name]))
            random_body_list.append(Variable(batch[random_body_name]))

        if use_gpu:
            query_title, query_body = query_title.cuda(), query_body.cuda()
            similar_title, similar_body = similar_title.cuda(), similar_body.cuda()
            random_title_list = map(lambda item: item.cuda(), random_title_list)
            random_body_list = map(lambda item: item.cuda(), random_body_list)
        
        optimizer.zero_grad()

        #query title
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_query_title = model(query_title)

        #query body
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_query_body = model(query_body)

        lstm_query = (lstm_query_title + lstm_query_body)/2.0

        #similar title
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_similar_title = model(similar_title)

        #similar body
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_similar_body = model(similar_body)

        lstm_similar = (lstm_similar_title + lstm_similar_body)/2.0

        lstm_random_list = []
        for ridx in range(len(random_title_list)):
            #random title
            model.hidden = model.init_hidden() 
            if use_gpu:
                model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
            lstm_random_title = model(random_title_list[ridx])

            #random body
            model.hidden = model.init_hidden() 
            if use_gpu:
                model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
            lstm_random_body = model(random_body_list[ridx])

            lstm_random = (lstm_random_title + lstm_random_body)/2.0
            lstm_random_list.append(lstm_random)
        #end for
           
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        score_pos = cosine_similarity(lstm_query, lstm_similar)

        score_list = []
        score_list.append(score_pos)
        for ridx in range(len(lstm_random_list)):
            score_neg = cosine_similarity(lstm_query, lstm_random_list[ridx])
            score_list.append(score_neg)

        X_scores = torch.stack(score_list, 1) #[batch_size, K=101]
        y_targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor)) #[batch_size]
        if use_gpu:
            y_targets = y_targets.cuda()
        loss = criterion(X_scores, y_targets) #y_target=0
        loss.backward()
        optimizer.step()
                
        running_train_loss += loss.cpu().data[0]        
        
    #end for
    training_loss.append(running_train_loss)
    learning_rate_schedule.append(scheduler.get_lr())
    print "epoch: %4d, training loss: %.4f" %(epoch+1, running_train_loss)
    
    torch.save(model, SAVE_PATH + SAVE_NAME)

    #early stopping
    patience = 4
    min_delta = 0.1
    if epoch == 0:
        patience_cnt = 0
    elif epoch > 0 and training_loss[epoch-1] - training_loss[epoch] > min_delta:
        patience_cnt = 0
    else:
        patience_cnt += 1

    if patience_cnt > patience:
        print "early stopping..."
        break

#end for
"""

print "loading pre-trained model..."
model = torch.load(SAVE_PATH)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()

"""

print "scoring test questions..."
running_test_loss = 0.0

test_data_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4, 
    drop_last = True)
        
model.eval()

for batch in tqdm(test_data_loader):

    query_idx = batch['query_idx']
    query_title = Variable(batch['query_title'])
    query_body = Variable(batch['query_body'])
    similar_title = Variable(batch['similar_title'])
    similar_body = Variable(batch['similar_body'])

    random_title_list = []
    random_body_list = []
    for ridx in range(20): #number of retrieved (bm25) examples
        random_title_name = 'random_title_' + str(ridx)
        random_body_name = 'random_body_' + str(ridx)
        random_title_list.append(Variable(batch[random_title_name]))
        random_body_list.append(Variable(batch[random_body_name]))

    if use_gpu:
        query_title, query_body = query_title.cuda(), query_body.cuda()
        similar_title, similar_body = similar_title.cuda(), similar_body.cuda()
        random_title_list = map(lambda item: item.cuda(), random_title_list)
        random_body_list = map(lambda item: item.cuda(), random_body_list)
    
    #query title
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_query_title = model(query_title)

    #query body
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_query_body = model(query_body)

    lstm_query = (lstm_query_title + lstm_query_body)/2.0

    #similar title
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_similar_title = model(similar_title)

    #similar body
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_similar_body = model(similar_body)

    lstm_similar = (lstm_similar_title + lstm_similar_body)/2.0

    lstm_random_list = []
    for ridx in range(len(random_title_list)):
        #random title
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_random_title = model(random_title_list[ridx])

        #random body
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_random_body = model(random_body_list[ridx])

        lstm_random = (lstm_random_title + lstm_random_body)/2.0
        lstm_random_list.append(lstm_random)
    #end for
           
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    score_pos = cosine_similarity(lstm_query, lstm_similar)

    score_list = []
    score_list.append(score_pos)
    for ridx in range(len(lstm_random_list)):
        score_neg = cosine_similarity(lstm_query, lstm_random_list[ridx])
        score_list.append(score_neg)

    X_scores = torch.stack(score_list, 1) #[batch_size, K=101]
    y_targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor)) #[batch_size]
    if use_gpu:
        y_targets = y_targets.cuda()
    loss = criterion(X_scores, y_targets) #y_target=0
    running_test_loss += loss.cpu().data[0]        
    
    #save scores to data frame
    lstm_query_idx = query_idx.cpu().numpy()
    lstm_retrieved_scores = X_scores.cpu().data.numpy()[:,1:] #skip positive score
    for row, qidx in enumerate(lstm_query_idx):
        test_idx_df.loc[test_idx_df['query_id'] == qidx, 'lstm_score'] = " ".join(lstm_retrieved_scores[row,:].astype('str'))
#end for        
    
print "total test loss: ", running_test_loss
print "number of NaN: ", test_idx_df.isnull().sum()
test_idx_df = test_idx_df.dropna() #NaNs are due to restriction: range(100)

print "computing ranking metrics..."
lstm_mrr_test = compute_mrr(test_idx_df, score_name='lstm_score')
print "lstm MRR (test): ", np.mean(lstm_mrr_test)

lstm_pr1_test = precision_at_k(test_idx_df, K=1, score_name='lstm_score')
print "lstm P@1 (test): ", np.mean(lstm_pr1_test)

lstm_pr5_test = precision_at_k(test_idx_df, K=5, score_name='lstm_score')
print "lstm P@5 (test): ", np.mean(lstm_pr5_test)

lstm_map_test = compute_map(test_idx_df, score_name='lstm_score')
print "lstm map (test): ", np.mean(lstm_map_test)


#generate plots
plt.figure()
plt.plot(training_loss, label='Adam')
plt.title("LSTM Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('./training_loss.png')

plt.figure()
plt.plot(learning_rate_schedule, label='learning rate')
plt.title("LSTM learning rate schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.savefig('./lstm_learning_rate_schedule.png')
        
            
#save for plotting
figures_lstm = {}

figures_lstm['lstm_mrr_test'] = [np.mean(lstm_mrr_test)]
figures_lstm['lstm_pr1_test'] = [np.mean(lstm_pr1_test)]
figures_lstm['lstm_pr5_test'] = [np.mean(lstm_pr5_test)]
figures_lstm['lstm_map_test'] = [np.mean(lstm_map_test)]

figures_lstm['lstm_training_loss'] = training_loss
figures_lstm['lstm_learning_rate'] = learning_rate_schedule 

filename = SAVE_PATH + '/figures_lstm.dat' 
with open(filename, 'w') as f:
    pickle.dump(figures_lstm, f)



