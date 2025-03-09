# Fine-tuning ModernBERT for Text Classification with PyTorch Lightning

In this technical guide, we'll explore how to fine-tune ModernBERT for text classification tasks using PyTorch Lightning. We'll cover the implementation details, model architecture, and best practices for training and evaluation.

## Understanding the Architecture

Our implementation leverages the power of ModernBERT's pre-trained language model combined with PyTorch Lightning's organized training structure. Here's what makes our implementation special:

1. **ModernBERT Base**: We use the `answerdotai/ModernBERT-base` model as our backbone, which provides powerful contextual representations. This model is loaded using the Hugging Face `AutoModelForSequenceClassification` class, which automatically handles the appropriate model architecture.

2. **Classification Head**: The `AutoModelForSequenceClassification` adds a classification layer on top of BERT's [CLS] token output. This is configured with the appropriate number of output classes via the `num_labels` parameter.

3. **PyTorch Lightning Structure**: Our implementation uses PyTorch Lightning's `LightningModule` class, which organizes the code into well-defined methods:
   - `forward()`: Processes input_ids and attention_mask through the model and returns logits
   - `training_step()`, `validation_step()`, `test_step()`: Handle the training, validation, and testing loops
   - `configure_optimizers()`: Sets up the AdamW optimizer with an appropriate learning rate

4. **Metrics Tracking**: We use torchmetrics' `Accuracy` class to track performance metrics during training, validation, and testing phases. These metrics are automatically logged and can be visualized.

5. **Tokenization and Input Processing**: The model expects tokenized inputs with `input_ids` and `attention_mask` tensors, which are processed by the BERT model to generate contextual embeddings.

## Implementation Details

### Setting Up the Environment

First, let's set up our development environment using Poetry for dependency management:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone and set up the project
git clone <repository-url>
cd finetuning_bert

# Install dependencies
poetry install
poetry shell
```

### Model Implementation

Let's break down the key components of our `BertClassifier` implementation:

```python
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from torchmetrics import Accuracy

class BertClassifier(pl.LightningModule):
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
```

### Data Processing

We use a custom `TextClassificationDataModule` for efficient data handling:

```python
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

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
```

### Training Pipeline

Here's how to set up the training pipeline:

```python
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Prepare your dataset
train_df, val_df, test_df = prepare_data('sample_dataset.csv')
data_module = TextClassificationDataModule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    batch_size=16
)

# Initialize model
model = BertClassifier(
    n_classes=2,  # Adjust based on your task
    learning_rate=2e-5
)

# Configure trainer
trainer = pl.Trainer(
    max_epochs=3,
    accelerator='auto',  # Automatically detect hardware
    devices=1,
    enable_progress_bar=True,
    enable_model_summary=True
)

# Train the model
trainer.fit(model, data_module)
```

## Best Practices and Tips

1. **Learning Rate**: Start with a small learning rate (2e-5 to 5e-5) to prevent catastrophic forgetting.
2. **Batch Size**: Use the largest batch size that fits in your GPU memory (typically 16-32).
3. **Gradient Clipping**: Consider adding gradient clipping to prevent exploding gradients:
```python
trainer = pl.Trainer(
    gradient_clip_val=1.0,
    # other parameters...
)
```

## Monitoring Training Progress

PyTorch Lightning automatically logs these metrics during training:

- Training Loss
- Training Accuracy
- Validation Loss
- Validation Accuracy

You can visualize these metrics using TensorBoard, which is automatically set up by PyTorch Lightning. The logs are stored in the 'lightning_logs' directory by default:

```python
trainer = pl.Trainer(
    max_epochs=3,
    accelerator='auto',
    devices=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    # TensorBoard logging is enabled by default
)
```

## Model Evaluation

Evaluate your model on the test set:

```python
# Run evaluation
results = trainer.test(model, data_module)
print(f"Test Accuracy: {results[0]['test_acc']:.4f}")
```

## Saving and Loading Models

```python
# Save the model and tokenizer
model.save_model(os.path.join('saved_model', 'model'))  # This saves both checkpoint and config
data_module.tokenizer.save_pretrained(os.path.join('saved_model', 'tokenizer'))

# Load the model
model = BertClassifier.load_from_checkpoint(
    os.path.join('saved_model', 'model.ckpt'),
    n_classes=2  # Make sure this matches your training configuration
)
```

## Inference with the Trained Model

Once you've trained and saved your model, you can use it for inference on new text data. Our implementation provides a convenient `TextClassifier` class that handles all the necessary steps:

```python
import torch
from transformers import AutoTokenizer
from model import BertClassifier
import os

class TextClassifier:
    def __init__(self, model_path='saved_model'):
        # Load the saved model and tokenizer
        self.model = BertClassifier.load_from_checkpoint(
            os.path.join(model_path, 'model.ckpt'),
            n_classes=2  # Make sure this matches your training configuration
        )
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
        self.model.eval()  # Set the model to evaluation mode
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def preprocess_text(self, text, max_length=128):
        # Tokenize the input text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    @torch.no_grad()
    def predict(self, text):
        # Preprocess the input text
        inputs = self.preprocess_text(text)
        
        # Get model predictions
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        return {
            'prediction': prediction.item(),
            'probabilities': probabilities[0].tolist()
        }
```

### Text Preprocessing

The `TextClassifier` includes a method to preprocess text inputs, handling tokenization and preparing the tensors for the model:

```python
def preprocess_text(self, text, max_length=128):
    # Tokenize the input text
    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'].to(self.device),
        'attention_mask': encoding['attention_mask'].to(self.device)
    }
```

### Making Predictions

The `predict` method handles the entire inference pipeline, from preprocessing to returning predictions and probabilities:

```python
@torch.no_grad()
def predict(self, text):
    # Preprocess the input text
    inputs = self.preprocess_text(text)
    
    # Get model predictions
    outputs = self.model(**inputs)
    probabilities = torch.softmax(outputs, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    
    return {
        'prediction': prediction.item(),
        'probabilities': probabilities[0].tolist()
    }
```

### Example Usage

Here's how to use the `TextClassifier` for inference:

```python
# Initialize the classifier
classifier = TextClassifier()

# Example text for prediction
text = "I really like the movie I watched last night."

# Get prediction
result = classifier.predict(text)
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")
```

The `prediction` value corresponds to the class index (e.g., 0 for negative, 1 for positive in a binary sentiment classification task), while `probabilities` contains the confidence scores for each class.

## Conclusion

Fine-tuning ModernBERT using PyTorch Lightning provides a clean and efficient way to create powerful text classifiers. The combination of BERT's pre-trained knowledge and PyTorch Lightning's organized training structure makes it easier to implement and experiment with different configurations.

Key takeaways:
- Use Poetry for clean dependency management
- Leverage PyTorch Lightning's built-in features for organized training
- Monitor training metrics for optimal performance
- Follow best practices for learning rates and batch sizes
- Implement a robust inference pipeline for real-world applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.