import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroscedasticKLLoss:
    """
    Heteroscedastic loss combining Gaussian NLL, KL divergence penalty, and BCE loss.
    
    This loss function implements:
    1. Gaussian Negative Log-Likelihood for continuous features (mean + variance prediction)
    2. KL divergence penalty to regularize predicted variance toward empirical variance
    3. Binary Cross-Entropy loss for binary flag features
    4. Feature-specific weighting for KL regularization
    """
    
    def __init__(self, empirical_var, kl_feature_weights, cont_idx, flag_idx, device):
        """
        Initialize the loss function.
        
        Args:
            empirical_var (torch.Tensor): Empirical variance of continuous features [num_continuous_features]
            kl_feature_weights (torch.Tensor): Feature-specific weights for KL penalty [num_continuous_features]
            cont_idx (list): Indices of continuous features in target tensor
            flag_idx (list): Indices of binary flag features in target tensor  
            device (torch.device): Device to place tensors on
        """
        self.empirical_var = empirical_var.to(device)
        self.kl_feature_weights = kl_feature_weights.to(device)
        self.cont_idx = cont_idx
        self.flag_idx = flag_idx
        self.device = device
    
    def forward(self, mean, logvar, logits, targets, kl_weight):
        """
        Compute heteroscedastic loss.

        heteroscedastic loss = NLL + kl_weight * KL + BCE
        NLL: Gaussian Negative Log-Likelihood: fit mean and variance to targets
        KL: KL divergence penalty: how much the predicted variance deviates from empirical variance
        BCE: Binary Cross-Entropy Loss: for flag features
        
        Args:
            mean (torch.Tensor): Predicted means for continuous features [batch_size, seq_len, num_continuous]
            logvar (torch.Tensor): Predicted log-variance for continuous features [batch_size, seq_len, num_continuous]
            logits (torch.Tensor): Predicted logits for binary flags [batch_size, seq_len, num_flags]
            targets (list): List of target tensors, one per sequence in batch [seq_len, num_features]
            kl_weight (float): Weight for KL divergence penalty (annealed during training)
            
        Returns:
            dict: Dictionary containing total loss and component losses
        """
        total_loss = 0.0
        nll_loss_sum = 0.0
        kl_loss_sum = 0.0
        bce_loss_sum = 0.0
        valid_count = 0
        
        for j, target in enumerate(targets):
            L = min(mean.shape[1], target.shape[0])
            
            # Create mask for valid (non-NaN) continuous features
            valid_mask = ~torch.isnan(target[:L, self.cont_idx])  # [L, num_continuous]
            
            if valid_mask.sum() == 0:
                continue  # Skip if no valid data in this sequence
            
            # 1. Gaussian Negative Log-Likelihood Loss for continuous features
            # NLL = 0.5 * [log(var) + (target - mean)^2 / var] 
            # Use wider clamping: [-1.5, 1.5] -> var in [0.22, 4.48]
            # Need wider range to prevent mean_head gradient explosion from squared_error/var
            #stop clamping altogether, we wanna make sure we are learning proper variance.
            #logvar_clamped = torch.clamp(logvar[j, :L, :], min=-1, max=1)
            logvar_clean = torch.clamp(logvar[j, :L, :], min=-3.0, max=3.0)
            
            # Debug: Check for extreme logvar values that could cause numerical issues
            if logvar_clean.abs().max() > 8.0:
                print(f"  DEBUG: Extreme logvar range: [{logvar_clean.min():.2f}, {logvar_clean.max():.2f}]")
            
            # Use exp with larger epsilon for stability
            var = torch.exp(logvar_clean) + 1e-3  # Larger epsilon prevents division issues with extreme logvar
            
            # Additional safety: check for any extreme values
            if torch.isnan(logvar_clean).any() or torch.isinf(logvar_clean).any():
                print(f"  WARNING: Invalid logvar_clean detected, skipping sequence {j}")
                continue
            if torch.isnan(var).any() or torch.isinf(var).any():
                print(f"  WARNING: Invalid var detected, skipping sequence {j}")
                continue
            
            # Compute squared error, masking NaN positions
            squared_error = (target[:L, self.cont_idx] - mean[j, :L, :]) ** 2
            squared_error = torch.where(valid_mask, squared_error, torch.zeros_like(squared_error))
            
            # Compute NLL loss with additional safety
            nll = 0.5 * (logvar_clean + squared_error / var)
            nll_loss = torch.where(valid_mask, nll, torch.zeros_like(nll)).sum()
            
            # Safety check on NLL loss
            if torch.isnan(nll_loss) or torch.isinf(nll_loss):
                print(f"  WARNING: Invalid NLL loss detected, skipping sequence {j}")
                continue
            
            # 2. KL Divergence Penalty - Ultra Conservative, been havving issues
            # KL(q||p) = 0.5 * [-log(var_q) + log(var_p) + var_q/var_p - 1]
            # Use very safe empirical variance bounds
            #dont think this is really used anymore but kept for saftey
            safe_emp_var = torch.clamp(self.empirical_var, min=0.6, max=1.8)  # Tighter bounds
            
            # Compute KL terms with extreme safety checks
            log_emp_var = torch.log(safe_emp_var)
            var_ratio = torch.clamp(var / safe_emp_var, min=0.2, max=5.0)  # Much tighter ratio
            
            # Compute KL with bounds checking
            kl_div = 0.5 * (-logvar_clean + log_emp_var + var_ratio - 1.0)
            
            # Safety check on KL divergence
            if torch.isnan(kl_div).any() or torch.isinf(kl_div).any():
                print(f"  WARNING: Invalid KL divergence detected, skipping sequence {j}")
                continue
                
            # Apply feature-specific weights and mask NaN positions
            kl_loss = torch.where(valid_mask, kl_div * self.kl_feature_weights, torch.zeros_like(kl_div)).sum()
            
            # Safety check on KL loss
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                print(f"  WARNING: Invalid KL loss detected, skipping sequence {j}")
                continue
            
            # 3. Binary Cross-Entropy Loss for flag features
            bce_loss = F.binary_cross_entropy_with_logits(
                logits[j, :L, :], 
                target[:L, self.flag_idx].float(), 
                reduction="sum"
            )
            
            # Accumulate losses
            nll_loss_sum += nll_loss
            kl_loss_sum += kl_loss
            bce_loss_sum += bce_loss
            valid_count += valid_mask.sum().item()
        
        # Normalize by valid count and combine losses
        if valid_count > 0:
            nll_loss_avg = nll_loss_sum / valid_count
            kl_loss_avg = kl_loss_sum / valid_count  
            bce_loss_avg = bce_loss_sum / valid_count
            total_loss = nll_loss_avg + bce_loss_avg + kl_weight * kl_loss_avg
        else:
            # No valid data in batch
            nll_loss_avg = torch.tensor(0.0, device=self.device)
            kl_loss_avg = torch.tensor(0.0, device=self.device)
            bce_loss_avg = torch.tensor(0.0, device=self.device)
            total_loss = torch.tensor(0.0, device=self.device)
        
        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss_avg,
            'kl_loss': kl_loss_avg,
            'bce_loss': bce_loss_avg,
            'valid_count': valid_count
        }
    
    def compute_mae(self, mean, targets):
        """
        Compute Mean Absolute Error on continuous features for interpretability.
        
        Args:
            mean (torch.Tensor): Predicted means [batch_size, seq_len, num_continuous]
            targets (list): List of target tensors [seq_len, num_features]
            
        Returns:
            torch.Tensor: MAE averaged over valid positions
        """
        mae_sum = 0.0
        valid_count = 0
        
        for j, target in enumerate(targets):
            L = min(mean.shape[1], target.shape[0])
            
            # Create mask for valid positions, any nan should have been dropped but for safety
            valid_mask = ~torch.isnan(target[:L, self.cont_idx])
            
            if valid_mask.sum() == 0:
                continue
                
            # Compute absolute error
            ae = torch.where(
                valid_mask, 
                (mean[j, :L, :] - target[:L, self.cont_idx]).abs(), 
                torch.zeros_like(mean[j, :L, :])
            )
            
            mae_sum += ae.sum().item()
            valid_count += valid_mask.sum().item()
        
        return mae_sum / max(1, valid_count)