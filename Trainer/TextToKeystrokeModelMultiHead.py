
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
import sys

class TextToKeystrokeModelMultiHead(nn.Module):

    def __init__(self, base_model, num_continuous=3, num_flags=7):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU()
        )
        
        # Heteroscedastic regression heads - predict mean AND variance
        self.mean_head = nn.Linear(256, num_continuous)
        self.logvar_head = nn.Linear(256, num_continuous)  # log-variance for numerical stability
        
        # Initialize logvar head to predict small variance initially
        #small random weights instead of getting stuck in constant output
        nn.init.normal_(self.logvar_head.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.logvar_head.bias, 0.0)
        #old values"
        #nn.init.constant_(self.logvar_head.weight, 0.0)
        #nn.init.constant_(self.logvar_head.bias, -0.5) 
        
        # Classification head (binary flags)
        self.classification_head = nn.Linear(256, num_flags)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = x.last_hidden_state  # [B, T, hidden]
        
        shared = self.backbone(hidden)  # [B, T, 256]
        
        mean = self.mean_head(shared)  # [B, T, num_continuous]
        logvar = self.logvar_head(shared)  # [B, T, num_continuous]
        logits = self.classification_head(shared)  # [B, T, num_flags]
        return mean, logvar, logits