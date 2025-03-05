import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import AutoModelForSequenceClassification
from torchmetrics import Accuracy

class BertClassifier(LightningModule):
    def __init__(self, n_classes: int, learning_rate: float = 2e-5):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('answerdotai/ModernBERT-base', num_labels=n_classes)
        self.learning_rate = learning_rate
        
        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=n_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=n_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, labels)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def save_model(self, path):
        # Save the model checkpoint
        self.trainer.save_checkpoint(f"{path}.ckpt")
        # Save the BERT model configuration
        self.model.config.save_pretrained(path)