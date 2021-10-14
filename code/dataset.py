import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer

from keyword2vec import KRLawData

class KRLawDataset(Dataset):
    """ Dataloader for Korean Law Dataset.
        blank classification model을 통과시킨 결과를 저장.
    """
    def __init__(self, data: pd.DataFrame, args):
        super().__init__()
        
        self.data = data
        self.args = args
        self.keyword2vec = pd.read_csv(args.keyword2vec)
        self.keyword2vec.set_index(['키워드'], drop=True, inplace=True)
        
        self.tokenizer1 = AutoTokenizer.from_pretrained(args.electra_model)
        self.tokenizer2 = BertTokenizer.from_pretrained(args.bert_model)

        if args.blank_cls_model != "None":
            self.blank_model = AutoModelForSequenceClassification.from_pretrained(args.blank_cls_model)

        self.max_token_len = args.max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data_row = self.data.iloc[idx]

        # input 문장
        contents = data_row.contents

        text_list = contents.split("[SEP]")

        blank_cls_list = []
        if self.args.blank_cls_model != "None":
            for text in text_list:
                tmp_inputs = self.tokenizer1(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
                tmp_input_ids = tmp_inputs.input_ids
                tmp_mask = tmp_inputs.attention_mask
                blank_cls_list.append(self.blank_model(input_ids=tmp_input_ids, attention_mask=tmp_mask).logits.argmax(-1).item())

        if any(blank_cls_list):
            if_label = torch.FloatTensor([1])
        else:
            if_label = torch.FloatTensor([0])

        # input 문장에 해당하는 키워드 벡터 저장
        keyword_vector = torch.FloatTensor(KRLawData.get_keyword_vector_for_each_text(contents, self.keyword2vec))
        
        # onehot label
        labels = torch.FloatTensor(eval(data_row.label_onehot))

        encoding = self.tokenizer2.encode_plus(
            contents,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].flatten()
        mask = encoding['attention_mask'].flatten()
        
        return dict(contents=contents, if_label=if_label, keyword_vector=keyword_vector, input_ids=input_ids, attention_mask=mask, labels=labels)