from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextClassificationDataModule(LightningDataModule):
    def __init__(self, train_df=None, val_df=None, test_df=None, 
                 tokenizer_name='bert-base-uncased', batch_size=16, max_length=128):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            if self.train_df is not None:
                self.train_dataset = TextDataset(
                    texts=self.train_df['text'].values,
                    labels=self.train_df['label'].values,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length
                )
            if self.val_df is not None:
                self.val_dataset = TextDataset(
                    texts=self.val_df['text'].values,
                    labels=self.val_df['label'].values,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length
                )

        if stage == 'test' or stage is None:
            if self.test_df is not None:
                self.test_dataset = TextDataset(
                    texts=self.test_df['text'].values,
                    labels=self.test_df['label'].values,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )