import pandas as pd
import re
import numpy as np
import argparse

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import KRLawDataset
from keyword2vec import KRLawData

class KRLawModel(pl.LightningModule):
    def __init__(self, args, n_training_steps=None, n_warmup_steps=None):
        super().__init__()

        self.args = args
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.bert = BertModel.from_pretrained(args.bert_model, return_dict=True)

        self.kw2vec_df = pd.read_csv(args.keyword2vec)
        self.kw2vec_df.set_index(['키워드'], drop=True, inplace=True)

        self.classifier = nn.Linear(len(self.kw2vec_df.columns) + self.bert.config.hidden_size, args.n_class)

        self.criterion = nn.BCELoss()
        
    def forward(self, if_label, keyword_vector, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size()[0]
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_outputs.last_hidden_state

        if self.args.mode == "CLS":
            tmp_state = last_hidden_state[:, 0]
            concat_state = torch.cat([keyword_vector, tmp_state], dim=1)

        # CLS 토큰과 SEP 토큰의 last hidden state의 max pooling 사용
        elif self.args.mode == "CLS+SEP":
            special_token_idx = [[0] for _ in range(batch_size)]
            sep_pos_tensor = (input_ids == 3).nonzero()
            sep_pos_list = sep_pos_tensor.tolist()
            for i, j in sep_pos_list:
                special_token_idx[i].append(j)
            
            # special_token_idx example => [[0, 25], [0, 24], [0, 125], [0, 31]]
            for pos_i, pos_list in enumerate(special_token_idx):
                for j, pos_j in enumerate(pos_list):
                    if j == 0:
                        tmp_state = last_hidden_state[pos_i, pos_j].unsqueeze(0)
                        continue
                    tmp_state = torch.cat([tmp_state, last_hidden_state[pos_i, pos_j].unsqueeze(0)], dim=0)
                # max pooling
                tmp_state = torch.max(tmp_state, dim=0)[0]
            
                if pos_i == 0:
                    ret_state = tmp_state.unsqueeze(0)
                    continue
                ret_state = torch.cat([ret_state, tmp_state.unsqueeze(0)], dim=0)   # ret_state shape = [batch size, hidden size]
            
            concat_state = torch.cat([keyword_vector, ret_state], dim=1)

        output = self.classifier(concat_state)
        output = torch.sigmoid(output)

        # blank classification model 이용
        if self.args.blank_cls_model != "None":
            output = output * if_label

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        if_label = batch["if_label"]
        keyword_vector = batch["keyword_vector"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(if_label, keyword_vector, input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        if_label = batch["if_label"]
        keyword_vector = batch["keyword_vector"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(if_label, keyword_vector, input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if_label = batch["if_label"]
        keyword_vector = batch["keyword_vector"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(if_label, keyword_vector, input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        
        for i in range(self.args.n_class):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{str(i)}_roc_auc/Train", class_roc_auc, self.current_epoch)
   
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps)
     
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))