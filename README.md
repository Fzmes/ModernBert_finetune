# BERT Text Classification Model

This repository contains a PyTorch Lightning implementation of a BERT-based text classification model using the ModernBERT architecture. The model is designed for multi-class text classification tasks and includes features for training, validation, and testing.

## Features

- Built with PyTorch Lightning for efficient training
- Uses ModernBERT-base as the backbone model
- Supports multi-class classification
- Includes accuracy metrics for training, validation, and testing
- Configurable learning rate and number of classes
- Model checkpointing and saving capabilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd finetuning_bert
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Activate the Poetry virtual environment:
```bash
poetry shell
```

## Model Architecture

The model is implemented in `model.py` and consists of:
- A BERT-based encoder (ModernBERT-base)
- A classification head for multi-class prediction
- Training, validation, and testing loops with accuracy metrics
- AdamW optimizer with configurable learning rate

## Usage

### Training

1. Prepare your dataset in the following format:
```python
dataset = {
    'input_ids': tensor,  # Tokenized input sequences
    'attention_mask': tensor,  # Attention masks
    'labels': tensor  # Classification labels
}
```

2. Initialize and train the model:
```python
from model import BertClassifier

# Initialize the model
model = BertClassifier(
    n_classes=<number_of_classes>,
    learning_rate=2e-5  # Adjust as needed
)

# Train using PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu',  # Use 'cpu' if no GPU available
    devices=1
)

trainer.fit(model, train_dataloader, val_dataloader)
```

### Evaluation

Evaluate the model on test data:
```python
trainer.test(model, test_dataloader)
```

### Saving and Loading

Save the model:
```python
model.save_model('path/to/save')
```

## Model Configuration

You can configure the following parameters when initializing the model:

- `n_classes`: Number of output classes (required)
- `learning_rate`: Learning rate for the AdamW optimizer (default: 2e-5)

## Metrics

The model tracks the following metrics during training:
- Training Loss
- Training Accuracy
- Validation Loss
- Validation Accuracy
- Test Loss
- Test Accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.