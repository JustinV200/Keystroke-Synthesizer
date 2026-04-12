import torch
from transformers import AutoModel
import torch.nn as nn

class TextToKeystrokeModelMultiHead(nn.Module):
    """DeBERTa-based model that predicts keystroke timing distributions from text.

    Architecture:
        1. A pretrained DeBERTa encoder maps token sequences to contextual embeddings.
        2. Token embeddings are expanded to character-level via an index mapping.
        3. A shared backbone MLP projects embeddings to a lower dimension.
        4. Two parallel heads predict per-character **mean** and **log-variance**
           for each continuous feature (DwellTime, FlightTime, typing_speed).

    Args:
        base_model (str): HuggingFace model identifier for the encoder.
        num_continuous (int): Number of continuous features to predict.
    """

    def __init__(self, base_model, num_continuous=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden, 768), nn.LayerNorm(768), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(768, 256), nn.ReLU()
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

    def forward(self, input_ids, attention_mask, token_to_char_idx=None):
        """Run a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs ``[B, T_tok]``.
            attention_mask (torch.Tensor): Attention mask ``[B, T_tok]``.
            token_to_char_idx (torch.Tensor | None): Character-to-token index
                mapping ``[B, T_char]``.  If provided, token embeddings are
                expanded to character-level before the regression heads.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(mean, logvar)`` each of shape
                ``[B, T, num_continuous]`` where *T* is ``T_char`` if the mapping
                is provided, else ``T_tok``.
        """
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = x.last_hidden_state  # [B, T_tok, hidden]
        
        # Expand token-level embeddings to character-level if mapping provided
        if token_to_char_idx is not None:
            hidden = self._expand_to_chars(hidden, token_to_char_idx)  # [B, T_char, hidden]
        
        shared = self.backbone(hidden)  # [B, T, 256]
        
        mean = self.mean_head(shared)  # [B, T, num_continuous]
        logvar = self.logvar_head(shared)  # [B, T, num_continuous]
        return mean, logvar

    def _expand_to_chars(self, token_embeds, token_to_char_idx):
        """Expand token-level embeddings to character-level using index mapping.
        
        Each character position gets the embedding of the subword token that covers it.
        This gives us per-character predictions instead of per-token predictions.
        
        Args:
            token_embeds: [B, T_tok, hidden] - DeBERTa encoder output
            token_to_char_idx: [B, T_char] - for each char, which token index covers it
        Returns:
            char_embeds: [B, T_char, hidden]
        """
        safe_idx = token_to_char_idx.clamp(0, token_embeds.size(1) - 1)
        char_embeds = torch.gather(
            token_embeds, 1,
            safe_idx.unsqueeze(-1).expand(-1, -1, token_embeds.size(-1))
        )
        return char_embeds