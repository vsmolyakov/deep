import platform
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter('ignore')

class Config:
    NB_EPOCHS = 3
    LR = 1e-5
    MAX_LEN = 256
    N_SPLITS = 5
    TRAIN_BS = 16
    VALID_BS = 32
    BERT_MODEL = '../input/huggingface-bert/bert-large-uncased'
    FILE_NAME = '../input/commonlitreadabilityprize/train.csv'
    TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    scaler = GradScaler()

class BERTDataset(Dataset):
    def __init__(self, review, target=None, is_test=False):
        self.review = review
        self.target = target
        self.is_test = is_test
        self.tokenizer = Config.TOKENIZER
        self.max_len = Config.MAX_LEN
    
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, idx):
        review = str(self.review[idx])
        review = ' '.join(review.split())
        
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        
        if self.is_test:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
            }
        else:    
            targets = torch.tensor(self.target[idx], dtype=torch.float)
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
            }


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        valid_dataloader,
        device
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.loss_fn = self.yield_loss
        self.device = device
        
    def yield_loss(self, outputs, targets):
        """
        This is the loss function for this task
        """
        return torch.sqrt(nn.MSELoss()(outputs, targets))
    
    def get_model(self):
        
        return self.model
    
    def train_one_epoch(self):
        """
        This function trains the model for 1 epoch through all batches
        """
        prog_bar = tqdm(enumerate(self.train_data), total=len(self.train_data))
        self.model.train()
        with autocast():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis)
                
                loss = self.loss_fn(outputs, targets)
                prog_bar.set_description('loss: {:.2f}'.format(loss.item()))

                Config.scaler.scale(loss).backward()
                Config.scaler.step(self.optimizer)
                Config.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
    def valid_one_epoch(self):
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        prog_bar = tqdm(enumerate(self.valid_data), total=len(self.valid_data))
        self.model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for idx, inputs in prog_bar:
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                ttis = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis)
                all_targets.extend(targets.cpu().detach().numpy().tolist())
                all_predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                
        val_rmse_loss = np.sqrt(mean_squared_error(all_targets, all_predictions))
        print('Validation RMSE: {:.2f}'.format(val_rmse_loss))
        
        return val_rmse_loss

class BERT_BASE_UNCASED(nn.Module):
    def __init__(self):
        super(BERT_BASE_UNCASED, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(768, 1)
    
    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.out(output)
        return output

def yield_optimizer(model):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.003,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return transformers.AdamW(optimizer_parameters, lr=Config.LR)

# Training Code
if __name__ == '__main__':

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda:0')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')
    
    data = pd.read_csv(Config.FILE_NAME)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data[['excerpt', 'target']]
    
    # Do Kfolds training and cross validation
    kf = StratifiedKFold(n_splits=Config.N_SPLITS)
    nb_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, 'bins'] = pd.cut(data['target'], bins=nb_bins, labels=False)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=data, y=data['bins'].values)):
        print(f"{'='*40} Fold: {fold} {'='*40}")
        
        train_data = data.loc[train_idx]
        valid_data = data.loc[valid_idx]
        
        train_set = BERTDataset(
            review = train_data['excerpt'].values,
            target = train_data['target'].values
        )

        valid_set = BERTDataset(
            review = valid_data['excerpt'].values,
            target = valid_data['target'].values
        )
        
        train = DataLoader(
            train_set,
            batch_size = Config.TRAIN_BS,
            shuffle = True,
            num_workers=8
        )

        valid = DataLoader(
            valid_set,
            batch_size = Config.VALID_BS,
            shuffle = False,
            num_workers=8
        )
        
        model = BERT_BASE_UNCASED().to(DEVICE)
        nb_train_steps = int(len(train_data) / Config.TRAIN_BS * Config.NB_EPOCHS)
        optimizer = yield_optimizer(model)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=nb_train_steps
        )
        
        trainer = Trainer(model, optimizer, scheduler, train, valid, DEVICE)

        best_loss = 100
        for epoch in range(1, Config.NB_EPOCHS+1):
            print(f"\n{'-'*20} EPOCH: {epoch} {'-'*20}\n")

            # Train for 1 epoch
            trainer.train_one_epoch()

            # Validate for 1 epoch
            current_loss = trainer.valid_one_epoch()
        
            if current_loss < best_loss:
                print(f"Saving best model in this fold: {current_loss:.4f}")
                torch.save(trainer.get_model().state_dict(), f"bert_base_uncased_ep_{epoch}_fold_{fold}.pt")
                best_loss = current_loss
        
        print(f"Best RMSE in fold: {fold} was: {best_loss:.4f}")
        print(f"Final RMSE in fold: {fold} was: {current_loss:.4f}")
        
    
    #inference
    test_file = pd.read_csv(Config.TEST_FILE_NAME)
    
    test_data = BERTDataset(test_file['excerpt'].values, is_test=True)
    test_data = DataLoader(
        test_data,  
        batch_size=Config.TRAIN_BS,
        shuffle=False
    )

    model_name = "/kaggle/working/bert_base_uncased_ep_2_fold_4.pt"
    model = BERT_BASE_UNCASED()
    model.load_state_dict(torch.load(model_name))
    model = model.to(DEVICE)

    print("Doing Predictions...")
    all_preds = []
    prog = tqdm(test_data, total=len(test_data))
    for data in prog:
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        ttis = data['token_type_ids'].to(DEVICE, dtype=torch.long)

        outputs = model(ids=ids, mask=mask, token_type_ids=ttis)
        all_preds.extend(outputs.cpu().detach().numpy().tolist())
    
    all_preds = [x[0] for x in all_preds]    
    test_ids = test_file['id'].tolist()

    # Form the sample submission
    sub = pd.DataFrame()
    sub['id'] = test_ids
    sub['target'] = all_preds
    sub.to_csv("submission.csv", index=None)
