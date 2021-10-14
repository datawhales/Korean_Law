import pandas as pd
import re
import numpy as np
import argparse
import os

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import classification_report

from dataset import KRLawDataset
from model import KRLawModel
from keyword2vec import KRLawData

class KRLawDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.batch_size = args.batch_size
        self.max_token_len = args.max_token_len

        self.train_df = pd.read_csv(args.train_data)
        self.val_df = pd.read_csv(args.val_data)
        self.test_df = pd.read_csv(args.test_data)

        self.train_df.dropna(inplace=True)
        self.val_df.dropna(inplace=True)
        self.test_df.dropna(inplace=True)
    
    def setup(self, stage=None):
        self.train_dataset = KRLawDataset(self.train_df, args)
        self.val_dataset = KRLawDataset(self.val_df, args)                              
        self.test_dataset = KRLawDataset(self.test_df, args)
                                         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.args.num_workers)
                         
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.args.num_workers)

def evaluate(trained_model, test_dataset, log_filepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)
    
    predictions = []
    labels = []
    for item in tqdm(test_dataset):
        _, prediction = trained_model(
            item["if_label"].unsqueeze(dim=0).to(device),
            item["keyword_vector"].unsqueeze(dim=0).to(device),
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device)
        )

        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())

    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()
    
    # [0.1, 0.2, .., 0.9]
    threshold_list = np.arange(0, 1, 0.1)[1:]

    print("********** Accuracy per threshold **********")
    acc_per_threshold = dict()
    for i in threshold_list:
        acc = accuracy(predictions, labels, threshold=i)
        acc_per_threshold[i] = acc
        print(f"Threshold: {i}, Accuracy: {acc:.6f}")
   
    max_acc_threshold = sorted(acc_per_threshold.items(), key=lambda x: x[1], reverse=True)[0][0]
    print(f"********** Max Threshold: {max_acc_threshold} **********")

    print("********** AUROC per class **********")
    auroc_per_class = dict()
    for i, name in enumerate(krlawdata.key_list):
        try:
            tag_auroc = auroc(predictions[:, i], labels[:, i], pos_label=1)
            auroc_per_class[i] = tag_auroc
            print(f"{name}: {tag_auroc}")
        except:
            auroc_per_class[i] = 0
            print(f"{name}: 0")

    print("********** Classification Report **********")
    y_pred = predictions.numpy()
    y_true = labels.numpy()
    upper, lower = 1, 0
    y_pred = np.where(y_pred > max_acc_threshold, upper, lower)
    
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=krlawdata.key_list,
        zero_division=0,
        output_dict=True
    )
    print(classification_report(
        y_true,
        y_pred,
        target_names=krlawdata.key_list,
        zero_division=0
    ))

    # accuracy calculation
    correct = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[i][j] < max_acc_threshold:
                predictions[i][j] = 0
            else:
                predictions[i][j] = 1
        if predictions[i].tolist() == labels[i].tolist():
            correct += 1

    acc = correct / len(predictions)
    
    print(f"********** Test Accuracy: {acc:.6f} **********")    

    with open(log_filepath, 'w') as f:
        # write args
        f.write('********** Args **********\n')
        for k in list(vars(args).keys()):
            f.write(f"{k}: {vars(args)[k]}\n")

        f.write('\n********** Accuracy per threshold **********\n')
        for k, v in acc_per_threshold.items():
            f.write(str(k)[:3] + '\t' + str(v)[7:-1] + '\n')
        f.write('\n')
        f.write('********** Max Threshold: ' + str(max_acc_threshold) + ' **********\n')
        f.write('\n********** AUROC per class **********\n')
        for k, v in auroc_per_class.items():
            f.write(str(k) + '\t' + str(v)[7:-1] + '\n')
        f.write('\n')
        f.write('********** Classification Report **********\n')
        f.write('라벨이름\tprecision\trecall  \tf1-score\tsupport\n')
        for k, v in cls_report.items():
            f.write(str(k) + '\t')
            f.write(f"{v['precision']:.6f}\t{v['recall']:.6f}\t{v['f1-score']:.6f}\t{v['support']}" + '\n')
        f.write('\n')
        f.write(f"********** Test Accuracy: {acc:.6f} **********")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korean Law Multi-label Classification")

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="random seed number")
    
    parser.add_argument("--batch_size", dest="batch_size", type=int,
                        default=12, help="batch size")

    parser.add_argument("--n_class", dest="n_class", type=int,
                        default=20, help="number of classes")

    parser.add_argument("--num_workers", dest="num_workers", type=int,
                        default=16, help="num workers")

    parser.add_argument("--epochs", dest="epochs", type=int,
                        default=5, help="training epochs")

    parser.add_argument("--mode", dest="mode", type=str,
                        default="CLS+SEP", help="{CLS+SEP, CLS}")

    parser.add_argument("--train", action="store_true", help="if train or not")

    parser.add_argument("--electra_model", dest="electra_model", type=str,
                        default="monologg/koelectra-base-v3-discriminator", help="Electra model name")

    parser.add_argument("--bert_model", dest="bert_model", type=str,
                        default="snunlp/KR-Medium", help="Bert model name")

    parser.add_argument("--max_token_len", dest="max_token_len", type=int,
                        default=256, help="max token length")

    parser.add_argument("--train_data", dest="train_data", type=str,
                        default="../data/contracts_dataset_1006/train.csv", help="training data file path")

    parser.add_argument("--val_data", dest="val_data", type=str,
                        default="../data/contracts_dataset_1006/val.csv", help="validation data file path")

    parser.add_argument("--test_data", dest="test_data", type=str,
                        default="../data/contracts_dataset_1006/test.csv", help="test data file path")

    parser.add_argument("--keywords_list", dest="keywords_list", type=str,
                        default="../data/all_keywords.txt", help="keywords list txt file path")
    
    parser.add_argument("--keyword2vec", dest="keyword2vec", type=str,
                        default="../data/keyword2vec_without_opt10.csv", help="keyword embedding csv file path")

    parser.add_argument("--base_data", dest="base_data", type=str,
                        default="../data/contracts1-16_without_opt10.csv", help="base data for keyword vectors and blank classification")

    parser.add_argument("--data_input", dest="data_input", type=str,
                        default="../data/contracts_input_final_label_fixed.csv", help="data input for training")

    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="../ckpt/CLS+SEP", help="dir path to save ckpt")

    parser.add_argument("--log_file", dest="log_file", type=str,
                        default="../log/CLS+SEP.txt", help="result log file path")

    parser.add_argument("--blank_cls_model", dest="blank_cls_model", type=str,
                        default="None", help="blank classification model path. ex: ../blank_cls_model/one-koelectra1-16")

    args = parser.parse_args()

    print("--------args--------")
    for k in list(vars(args).keys()):
        print(f"{k}: {vars(args)[k]}")
    print("--------args--------")

    if not os.path.exists("../log"):
        os.mkdir("../log")

    if not os.path.exists("../ckpt"):
        os.mkdir("../ckpt")

    pl.seed_everything(args.seed)

    krlawdata = KRLawData(args)
    data_module = KRLawDataModule(args)

    if args.train:
        steps_per_epoch = len(data_module.train_df) // args.batch_size
        total_training_steps = steps_per_epoch * args.epochs
        warmup_steps = total_training_steps // 5

        model = KRLawModel(args, n_training_steps=total_training_steps, n_warmup_steps=warmup_steps)

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir,
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )

        logger = TensorBoardLogger("lightning_logs", name="KRLaw-cls")

        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

        trainer = pl.Trainer(
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=[early_stopping_callback],
            max_epochs=args.epochs,
            gpus=1,
            progress_bar_refresh_rate=30
        )

        trainer.fit(model, data_module)

        trainer.test()
        
        model.bert.save_pretrained(args.save_dir + '_bert')

    # evaluate
    else:
        test_dataset = KRLawDataset(data_module.test_df, args)

        trained_model = KRLawModel.load_from_checkpoint(os.path.join(args.save_dir, "best-checkpoint.ckpt"), args=args)
        
        trained_model.eval()
        trained_model.freeze()
        
        evaluate(trained_model, test_dataset, args.log_file)
