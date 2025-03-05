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

# Example usage
if __name__ == '__main__':
    # Initialize the classifier
    classifier = TextClassifier()
    
    # Example text for prediction
    text = "I really like the movie I watched last night."
    
    # Get prediction
    result = classifier.predict(text)
    print(f"Prediction: {result['prediction']}")
    print(f"Probabilities: {result['probabilities']}")