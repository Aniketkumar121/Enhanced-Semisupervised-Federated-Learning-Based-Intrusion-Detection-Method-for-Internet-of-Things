import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

class SSFL_IDS_CELoss(nn.Module):
    def __init__(self):
        super(SSFL_IDS_CELoss, self).__init__()

    def forward(self, pred_pro, target_tensor):
        # Apply log_softmax on predictions
        pred_pro = F.log_softmax(pred_pro, dim=1)
        out = -1 * pred_pro * target_tensor
        return out.sum() / len(pred_pro)  # Average loss across batch


def compute_arith_mean(params, grads_history):
    """
    Compute arithmetic mean of gradients with better safety checks.
    Args:
        params: model parameters
        grads_history: list of gradients from different iterations/clients
    """
    if not grads_history:
        return params
        
    # Handle nested lists of gradients
    if isinstance(grads_history[0], list):
        # Flatten the structure for processing
        flat_grads = []
        for grad_batch in grads_history:
            flat_grads.extend([g for g in grad_batch if g is not None])
        grads_history = flat_grads
    
    if not grads_history:  # Check again after flattening
        return params
        
    for param_idx, param in enumerate(params):
        if param.grad is None:
            param.grad = torch.zeros_like(param.data)
            continue
        
        # Get valid gradients for this parameter (non-empty and matching shape)
        valid_grads = []
        for grad in grads_history:
            if param_idx < len(grad) and grad[param_idx] is not None and grad[param_idx].numel() > 0:
                if grad[param_idx].shape == param.shape:
                    valid_grads.append(grad[param_idx].data)
        
        # Only compute mean if we have valid gradients
        if valid_grads:
            param.grad.data = torch.mean(torch.stack(valid_grads), dim=0)
    
    return params

def compute_geo_mean(params, grads_history, lr=0.001, eps=1e-8):
    """
    Compute geometric mean of gradients.
    Args:
        params: model parameters
        grads_history: list of gradients from different iterations/clients
        lr: learning rate
        eps: small constant to avoid numerical issues
    """
    if not grads_history:
        return params
        
    # Handle nested lists of gradients
    if isinstance(grads_history[0], list):
        # Flatten the structure for processing
        flat_grads = []
        for grad_batch in grads_history:
            flat_grads.extend([g for g in grad_batch if g is not None])
        grads_history = flat_grads
    
    if not grads_history:  # Check again after flattening
        return params
        
    for param_idx, param in enumerate(params):
        if param.grad is None:
            continue
            
        # Get valid gradients for this parameter
        valid_grads = [grad[param_idx].data for grad in grads_history 
                     if param_idx < len(grad) and grad[param_idx] is not None]
        
        if not valid_grads:
            continue
            
        # Compute geometric mean (via log-space for numerical stability)
        # Add small epsilon to ensure we can take log of positive values
        log_grads = torch.stack([torch.log(torch.abs(g) + eps) for g in valid_grads])
        signs = torch.stack([torch.sign(g) for g in valid_grads])
        mean_log_grad = torch.mean(log_grads, dim=0)
        mean_sign = torch.sign(torch.mean(signs, dim=0))
        
        # Convert back from log space and apply sign
        param.grad.data = mean_sign * torch.exp(mean_log_grad)
    
    return params

def compute_grad_variance(param_gradients):
    """
    Compute variance of gradients across parameters.
    Args:
        param_gradients: list of parameter gradients
    Returns:
        Dictionary mapping parameter index to its gradient variance
    """
    if not param_gradients or all(g is None for g in param_gradients):
        return {}
    
    variance_dict = {}
    
    for i, grad in enumerate(param_gradients):
        if grad is not None:
            # Compute mean squared value (second moment)
            variance_dict[f'layer_{i}'] = grad.detach().pow(2).mean()
    
    return variance_dict

def l2_between_dicts(dict1, dict2):
    """
    Calculate L2 distance between two dictionaries of tensors.
    """
    if not dict1 or not dict2:
        return 0.0
        
    l2_distance = 0.0
    for key in dict1:
        if key in dict2:
            l2_distance += torch.sum((dict1[key] - dict2[key]) ** 2).item()
    return l2_distance

def apply_fishr_loss(model, grads_variance_history, penalty_weight=0.1, round_idx=0, anneal_iters=50):
    """
    Apply Fishr loss to penalize gradient variance differences.
    
    Args:
        model: The model whose gradients are being analyzed
        grads_variance_history: List of gradient variance dictionaries from clients
        penalty_weight: Weight factor for the penalty term
        round_idx: Current round/iteration number
        anneal_iters: Number of iterations for penalty annealing
    Returns:
        Fishr loss tensor
    """
    if not grads_variance_history:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    # Average the gradient statistics across clients
    averaged_grad_stats = {}
    for key in grads_variance_history[0].keys():
        # Get all variances for this key across clients
        key_vars = [stats[key] for stats in grads_variance_history if key in stats]
        if key_vars:
            averaged_grad_stats[key] = torch.stack(key_vars).mean(dim=0)
    
    # Calculate penalty as L2 distance between each client's stats and the average
    fishr_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for client_stats in grads_variance_history:
        fishr_loss += torch.tensor(l2_between_dicts(client_stats, averaged_grad_stats),
                                 device=next(model.parameters()).device)
    
    # Apply annealing schedule
    penalty_weight = penalty_weight if round_idx >= anneal_iters else (
        penalty_weight * (round_idx / anneal_iters))
    
    return fishr_loss * penalty_weight