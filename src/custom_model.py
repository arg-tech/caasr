import torch
from transformers import GPT2ForSequenceClassification
import torch.nn as nn

class MicroAndMacroContextPT2ForTokenClassification(GPT2ForSequenceClassification):
    def __init__(self, config=None, position_embedding_dim=128, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.position_embedding_dim = position_embedding_dim
        hidden_size = config.hidden_size
        num_labels = 4
        
        self.classification = nn.Linear(hidden_size, 3)
        
        self.position_feedforward = nn.Sequential(
            nn.Linear(position_embedding_dim, position_embedding_dim),
            nn.ReLU(),
            nn.Linear(position_embedding_dim, position_embedding_dim)
        )
        
        extended_dim = hidden_size + position_embedding_dim
        self.classification_layer = nn.Linear(extended_dim, num_labels)

    def forward(self, token_ids=None, attention_mask=None, position_embeddings=None, labels=None):
        outputs = super().forward(token_ids, attention_mask=attention_mask, labels=labels)

        last_hidden_output = outputs.hidden_states[-1].mean(dim=1)
        position_outputs = self.position_feedforward(position_embeddings)
        concatenated_outputs = torch.cat((last_hidden_output, position_outputs), dim=1)
        classification_logits = self.classification_layer(concatenated_outputs)
        outputs.classification_logits = classification_logits
        
        return outputs

