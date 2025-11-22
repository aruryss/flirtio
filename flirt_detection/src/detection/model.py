"""
DistilBERT-based Flirt Detection Model
Wrapper for binary classification of flirtatious vs non-flirtatious text
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from typing import Dict, Optional, Tuple


class FlirtDetectionModel(nn.Module):
    """
    DistilBERT-based binary classifier for flirt detection.
    
    Architecture:
    - DistilBERT base model (66M parameters)
    - Dropout layer for regularization
    - Linear classification head
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        """
        Initialize the FlirtDetectionModel.
        
        Args:
            model_name: Pretrained DistilBERT model name
            num_labels: Number of output classes (2 for binary classification)
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT weights during training
        """
        super(FlirtDetectionModel, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load pretrained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            labels: Ground truth labels (batch_size,) - optional
        
        Returns:
            Dictionary containing:
                - logits: Raw predictions (batch_size, num_labels)
                - loss: Cross-entropy loss (if labels provided)
                - hidden_states: Last hidden state from DistilBERT
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token representation
        hidden_state = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(hidden_state)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_state
        }
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Probabilities for each class (batch_size, num_labels)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.softmax(outputs['logits'], dim=1)
        return probs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Predicted class labels (batch_size,)
        """
        probs = self.predict_proba(input_ids, attention_mask)
        return torch.argmax(probs, dim=1)
    
    def save_model(self, path: str):
        """Save model weights and config"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """Load model from saved weights"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            model_name=checkpoint['model_name'],
            num_labels=checkpoint['num_labels']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Model loaded from {path}")
        return model


class FlirtDetectionTokenizer:
    """Wrapper for DistilBERT tokenizer with preprocessing"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Pretrained model name
            max_length: Maximum sequence length
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def encode_batch(
        self,
        texts: list,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            return_tensors: Return format ('pt' for PyTorch)
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def encode_single(self, text: str, return_tensors: str = 'pt'):
        """Encode a single text"""
        return self.encode_batch([text], return_tensors)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in model.
    
    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total