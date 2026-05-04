"""
MLP-based transition learning for soft label post-processing.

Trains a small Multi-Layer Perceptron to learn the mapping from biased
distributions to unbiased distributions.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from postprocess.smoothing import (
    Dataset,
    DistDataset,
    Smoothing,
    smooth_counts,
)


class TransitionMLP(nn.Module):
    """
    Multi-Layer Perceptron for learning biased → unbiased distribution mapping.
    
    Input features (3K dimensions):
    - biased_probs (K): Smoothed biased probability distribution
    - one_hot_proposed (K): One-hot encoding of proposed class
    - transition_row (K): Row from transition matrix for proposed class
    
    Output:
    - predicted_probs (K): Predicted unbiased distribution
    """
    
    def __init__(
        self, 
        num_classes: int, 
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        input_dim = num_classes * 3  # biased + one_hot + transition
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3 * num_classes)
        
        Returns:
            Output probabilities of shape (batch_size, num_classes)
        """
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)


def prepare_mlp_data(
    map_id_biased_probs: DistDataset,
    map_id_unbiased_probs: DistDataset,
    map_id_proposed_class: Dict[str, str],
    transition_c: Dict[str, List[float]],
    classes: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare training data for MLP.
    
    Args:
        map_id_biased_probs: Biased probability distributions
        map_id_unbiased_probs: Ground truth unbiased distributions
        map_id_proposed_class: Proposed class per object
        transition_c: Transition matrix {class -> [probs]}
        classes: List of class names (defines ordering)
    
    Returns:
        X: Feature array (N, 3K)
        Y: Target array (N, K)
        obj_ids: List of object IDs (for mapping back)
    """
    X_list = []
    Y_list = []
    obj_ids = []
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    
    for obj_id in map_id_biased_probs:
        if obj_id not in map_id_unbiased_probs or obj_id not in map_id_proposed_class:
            continue
        
        biased = map_id_biased_probs[obj_id]
        unbiased = map_id_unbiased_probs[obj_id]
        proposed = map_id_proposed_class[obj_id]
        
        # Biased probs (K,)
        biased_vec = np.array([biased.get(c, 0.0) for c in classes], dtype=np.float32)
        
        # One-hot proposed (K,)
        one_hot = np.zeros(K, dtype=np.float32)
        if proposed in class_to_idx:
            one_hot[class_to_idx[proposed]] = 1.0
        
        # Transition row (K,)
        if proposed in transition_c:
            trans_row = np.array(transition_c[proposed], dtype=np.float32)
        else:
            trans_row = np.ones(K, dtype=np.float32) / K  # Uniform fallback
        
        # Concatenate features
        x = np.concatenate([biased_vec, one_hot, trans_row])
        
        # Target: unbiased probs (K,)
        y = np.array([unbiased.get(c, 0.0) for c in classes], dtype=np.float32)
        
        X_list.append(x)
        Y_list.append(y)
        obj_ids.append(obj_id)
    
    return np.stack(X_list), np.stack(Y_list), obj_ids


def train_mlp(
    map_id_biased_probs: DistDataset,
    map_id_unbiased_probs: DistDataset,
    map_id_proposed_class: Dict[str, str],
    transition_c: Dict[str, List[float]],
    classes: List[str],
    hidden_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 10,
    verbose: bool = True,
) -> Tuple[Optional["TransitionMLP"], Dict]:
    """
    Train MLP to predict unbiased distributions from biased inputs.
    
    Args:
        map_id_biased_probs: Biased probability distributions
        map_id_unbiased_probs: Ground truth unbiased distributions
        map_id_proposed_class: Proposed class per object
        transition_c: Transition matrix
        classes: List of class names
        hidden_dim: Hidden layer dimension
        epochs: Maximum training epochs
        batch_size: Training batch size
        lr: Learning rate
        val_split: Fraction of data for validation
        patience: Early stopping patience
        verbose: Print training progress
    
    Returns:
        (trained_model, training_info)
    """
    # Prepare data
    X, Y, obj_ids = prepare_mlp_data(
        map_id_biased_probs,
        map_id_unbiased_probs,
        map_id_proposed_class,
        transition_c,
        classes,
    )
    
    if len(X) < 10:
        if verbose:
            print(f"Warning: Too few samples ({len(X)}) for MLP training")
        return None, {"error": f"Too few samples: {len(X)}"}
    
    # Split into train/val
    n_val = max(1, int(len(X) * val_split))
    indices = np.random.permutation(len(X))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    num_classes = len(classes)
    model = TransitionMLP(num_classes, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # KL divergence loss
    def kl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        pred = pred.clamp(min=eps)
        target = target.clamp(min=eps)
        return (target * (target.log() - pred.log())).sum(dim=-1).mean()
    
    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    if verbose:
        print(f"\nTraining MLP: {len(X_train)} train, {len(X_val)} val samples")
        print(f"Input dim: {X.shape[1]}, Hidden: {hidden_dim}, Output: {num_classes}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = kl_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        
        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = kl_loss(val_pred, Y_val_t).item()
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if verbose:
        print(f"  Final: best_val_loss={best_val_loss:.4f}")
    
    training_info = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "best_val_loss": best_val_loss,
        "final_epoch": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    
    return model, training_info


def create_mlp_method(
    model: "TransitionMLP",
    map_id_proposed_class: Dict[str, str],
    transition_c: Dict[str, List[float]],
    classes: List[str],
    smoothing: Smoothing = Smoothing.EPSILON,
    alpha: float = 0.5,
    epsilon: float = 1e-10,
) -> Callable[[Dataset], DistDataset]:
    """
    Create a post-processing method function using the trained MLP.
    
    This returns a function with the same interface as other methods:
    method(map_id_counts) -> map_id_probs
    
    Args:
        model: Trained TransitionMLP
        map_id_proposed_class: Proposed class per object
        transition_c: Transition matrix
        classes: List of class names
        smoothing: Smoothing type for input
        alpha: Smoothing alpha
        epsilon: Smoothing epsilon
    
    Returns:
        Method function: Dataset -> DistDataset
    """
    if model is None:
        # Return identity if no model
        def fallback_method(map_id_counts: Dataset) -> DistDataset:
            return {
                obj_id: smooth_counts(counts, smoothing, classes, alpha, epsilon)
                for obj_id, counts in map_id_counts.items()
            }
        return fallback_method
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    
    def mlp_method(map_id_counts: Dataset) -> DistDataset:
        """Apply MLP post-processing to counts."""
        result = {}
        
        # Prepare batch input
        obj_ids = []
        X_list = []
        
        for obj_id, counts in map_id_counts.items():
            # Smooth to get biased probs
            biased = smooth_counts(counts, smoothing, classes, alpha, epsilon)
            proposed = map_id_proposed_class.get(obj_id, classes[0])
            
            # Biased probs
            biased_vec = np.array([biased.get(c, 0.0) for c in classes], dtype=np.float32)
            
            # One-hot proposed
            one_hot = np.zeros(K, dtype=np.float32)
            if proposed in class_to_idx:
                one_hot[class_to_idx[proposed]] = 1.0
            
            # Transition row
            if proposed in transition_c:
                trans_row = np.array(transition_c[proposed], dtype=np.float32)
            else:
                trans_row = np.ones(K, dtype=np.float32) / K
            
            x = np.concatenate([biased_vec, one_hot, trans_row])
            X_list.append(x)
            obj_ids.append(obj_id)
        
        if not obj_ids:
            return result
        
        # Run inference
        X_tensor = torch.from_numpy(np.stack(X_list))
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor).numpy()
        
        # Convert back to dict format
        for obj_id, pred in zip(obj_ids, preds):
            result[obj_id] = {c: float(pred[i]) for i, c in enumerate(classes)}
        
        return result
    
    return mlp_method


def get_mlp_method_for_tuning(
    train_biased: Dataset,
    train_unbiased: Dataset,
    train_proposed: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    smoothing: Smoothing = Smoothing.EPSILON,
    alpha: float = 0.5,
    epsilon: float = 1e-10,
    hidden_dim: int = 64,
    epochs: int = 100,
    verbose: bool = True,
) -> Optional[Tuple[str, Callable[[Dataset], DistDataset], Dict]]:
    """
    Train MLP and return as a method tuple for tuning.
    
    This is the main entry point for integrating MLP into run_tuning.py.
    
    Args:
        train_biased: Training biased counts
        train_unbiased: Training unbiased counts (ground truth)
        train_proposed: Training proposed classes
        classes: List of class names
        transition_c: Transition matrix
        smoothing: Smoothing type
        alpha: Smoothing alpha
        epsilon: Smoothing epsilon
        hidden_dim: MLP hidden dimension
        epochs: Training epochs
        verbose: Print training progress
    
    Returns:
        (name, method, params) tuple, or None if training fails
    """
    # Smooth counts to probs for training
    train_biased_probs = {
        k: smooth_counts(c, smoothing, classes, alpha, epsilon)
        for k, c in train_biased.items()
    }
    train_unbiased_probs = {
        k: smooth_counts(c, smoothing, classes, alpha, epsilon)
        for k, c in train_unbiased.items()
    }
    
    # Train model
    model, info = train_mlp(
        train_biased_probs,
        train_unbiased_probs,
        train_proposed,
        transition_c,
        classes,
        hidden_dim=hidden_dim,
        epochs=epochs,
        verbose=verbose,
    )
    
    if model is None:
        return None
    
    # Create method
    method = create_mlp_method(
        model,
        train_proposed,
        transition_c,
        classes,
        smoothing=smoothing,
        alpha=alpha,
        epsilon=epsilon,
    )
    
    # Build name and params
    smoothing_suffix = f"epsilon_{epsilon}" if smoothing == Smoothing.EPSILON else f"alpha_{alpha}"
    name = f"mlp_h{hidden_dim}_{smoothing_suffix}"
    
    params = {
        "method": "mlp",
        "hidden_dim": hidden_dim,
        "smoothing": smoothing.name,
        "alpha": alpha,
        "epsilon": epsilon,
        "train_info": {
            "best_val_loss": info.get("best_val_loss"),
            "final_epoch": info.get("final_epoch"),
        },
    }
    
    return name, method, params
