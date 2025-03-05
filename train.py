import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from model import BertClassifier
from data_module import TextClassificationDataModule
import os

def prepare_data(csv_path):
    # Load your dataset here
    # This is an example assuming you have a CSV with 'text' and 'label' columns
    df = pd.read_csv(csv_path)
    
    # Split the data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df

def main():
    # Training configuration
    MAX_EPOCHS = 3
    BATCH_SIZE = 16
    N_CLASSES = 2  # Change this according to your dataset
    MODEL_SAVE_PATH = 'saved_model'  # Directory to save the model
    
    # Create save directory if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Initialize data module
    # Use the sample dataset for training
    train_df, val_df, test_df = prepare_data('sample_dataset.csv')
    data_module = TextClassificationDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=BATCH_SIZE
    )
    
    # Initialize model
    model = BertClassifier(n_classes=N_CLASSES)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)
    
    # Save the model and tokenizer
    model.save_model(os.path.join(MODEL_SAVE_PATH, 'model'))
    data_module.tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, 'tokenizer'))

if __name__ == '__main__':
    main()