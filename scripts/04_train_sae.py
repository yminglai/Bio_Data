"""
Step 4: Train Supervised 6-Slot SAE
Exactly one latent per rule, with supervision to enforce 1-to-1 mapping.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import json

class ActivationDataset(Dataset):
    def __init__(self, activations_file):
        with open(activations_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'h': torch.from_numpy(item['h']).float(),
            'rule_idx': item['rule_idx'],
            'answer_tokens': item['answer_tokens'],
        }

class SupervisedSAE(nn.Module):
    """
    6-slot SAE with supervised slot selection.
    Each slot corresponds to exactly one rule.
    """
    def __init__(self, d_model, n_slots=6, vocab_size=50257):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model
        
        # Encoder: maps activations to slot logits
        self.encoder = nn.Linear(d_model, n_slots, bias=True)
        
        # Decoder: reconstructs activations from slot activations
        self.decoder = nn.Linear(n_slots, d_model, bias=True)
        
        # Value heads: predict answer tokens from active slot
        # Simplified: predict first token of answer
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 256),
                nn.ReLU(),
                nn.Linear(256, vocab_size)
            )
            for _ in range(n_slots)
        ])
        
        # Initialize decoder to be close to identity (via transpose of encoder)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, h, temperature=1.0, hard=False):
        """
        Args:
            h: [batch, d_model] - input activations
            temperature: Gumbel-Softmax temperature (lower = more one-hot)
            hard: If True, use hard one-hot (straight-through estimator)
        
        Returns:
            z: [batch, n_slots] - slot activations (soft or hard one-hot)
            h_recon: [batch, d_model] - reconstructed activations
        """
        # Encode to slot logits
        logits = self.encoder(h)  # [batch, n_slots]
        
        # Gumbel-Softmax: soft one-hot over slots
        z = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        
        # Decode back to activation space
        h_recon = self.decoder(z)
        
        return z, h_recon, logits
    
    def predict_values(self, z, rule_idx):
        """
        Predict answer token logits using the active slot's value head.
        
        Args:
            z: [batch, n_slots]
            rule_idx: [batch] - which rule this is
        
        Returns:
            logits: [batch, vocab_size]
        """
        batch_size = z.shape[0]
        device = z.device
        
        # For each sample, use the value head corresponding to its rule
        all_logits = []
        for i in range(batch_size):
            slot_val = z[i, rule_idx[i]].unsqueeze(0).unsqueeze(1)  # [1, 1]
            logits = self.value_heads[rule_idx[i]](slot_val)  # [1, vocab_size]
            all_logits.append(logits)
        
        return torch.cat(all_logits, dim=0)  # [batch, vocab_size]

def compute_loss(model, h, rule_idx, answer_tokens, temperature, lambda_recon=1.0, 
                 lambda_sparse=1e-3, lambda_align=1.0, lambda_indep=1e-2, lambda_value=0.5):
    """
    Combined loss for supervised SAE.
    """
    batch_size = h.shape[0]
    device = h.device
    
    # Forward pass
    z, h_recon, logits = model(h, temperature=temperature)
    
    # 1. Reconstruction loss
    L_recon = F.mse_loss(h_recon, h)
    
    # 2. Sparsity loss (encourage low entropy = one-hot)
    # Entropy of softmax distribution
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
    L_sparse = entropy  # Want to minimize entropy
    
    # 3. Alignment loss (supervised slot selection)
    # z should be one-hot with peak at rule_idx
    L_align = F.cross_entropy(logits, rule_idx)
    
    # 4. Independence loss (decorrelate slots across batch)
    # Covariance of z should be diagonal
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / batch_size
    # Off-diagonal elements should be zero
    off_diag = cov - torch.eye(model.n_slots, device=device) * cov
    L_indep = (off_diag ** 2).sum()
    
    # 5. Value prediction loss
    # Predict first answer token from active slot
    answer_first_tokens = torch.tensor(
        [tokens[0] if len(tokens) > 0 else 0 for tokens in answer_tokens],
        device=device
    )
    
    value_logits = model.predict_values(z, rule_idx)
    L_value = F.cross_entropy(value_logits, answer_first_tokens)
    
    # Total loss
    total_loss = (
        lambda_recon * L_recon +
        lambda_sparse * L_sparse +
        lambda_align * L_align +
        lambda_indep * L_indep +
        lambda_value * L_value
    )
    
    # Compute accuracy for monitoring
    slot_pred = logits.argmax(dim=-1)
    slot_acc = (slot_pred == rule_idx).float().mean()
    
    value_pred = value_logits.argmax(dim=-1)
    value_acc = (value_pred == answer_first_tokens).float().mean()
    
    return {
        'loss': total_loss,
        'L_recon': L_recon.item(),
        'L_sparse': L_sparse.item(),
        'L_align': L_align.item(),
        'L_indep': L_indep.item(),
        'L_value': L_value.item(),
        'slot_acc': slot_acc.item(),
        'value_acc': value_acc.item(),
    }

def collate_fn(batch):
    """Custom collate function to handle variable-length answer tokens."""
    h = torch.stack([item['h'] for item in batch])
    rule_idx = torch.tensor([item['rule_idx'] for item in batch])
    answer_tokens = [item['answer_tokens'] for item in batch]
    
    return {
        'h': h,
        'rule_idx': rule_idx,
        'answer_tokens': answer_tokens,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_file', type=str, default='data/activations/train_activations.pkl')
    parser.add_argument('--output_dir', type=str, default='models/sae_6slot')
    parser.add_argument('--n_slots', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temp_start', type=float, default=1.0)
    parser.add_argument('--temp_end', type=float, default=0.1)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_sparse', type=float, default=1e-3)
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_indep', type=float, default=1e-2)
    parser.add_argument('--lambda_value', type=float, default=0.5)
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading activations from {args.activation_file}")
    dataset = ActivationDataset(args.activation_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Get dimensionality from first sample
    d_model = dataset[0]['h'].shape[0]
    print(f"Activation dimension: {d_model}")
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SupervisedSAE(d_model=d_model, n_slots=args.n_slots)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Temperature annealing schedule (linear)
    def get_temperature(epoch):
        alpha = epoch / (start_epoch + args.epochs)
        return args.temp_start * (1 - alpha) + args.temp_end * alpha
    
    # Training loop
    print(f"Starting training from epoch {start_epoch}...")
    history = []
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_metrics = {
            'loss': 0, 'L_recon': 0, 'L_sparse': 0, 'L_align': 0,
            'L_indep': 0, 'L_value': 0, 'slot_acc': 0, 'value_acc': 0
        }
        
        temperature = get_temperature(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            h = batch['h'].to(device)
            rule_idx = batch['rule_idx'].to(device)
            answer_tokens = batch['answer_tokens']
            
            optimizer.zero_grad()
            
            metrics = compute_loss(
                model, h, rule_idx, answer_tokens, temperature,
                lambda_recon=args.lambda_recon,
                lambda_sparse=args.lambda_sparse,
                lambda_align=args.lambda_align,
                lambda_indep=args.lambda_indep,
                lambda_value=args.lambda_value
            )
            
            metrics['loss'].backward()
            optimizer.step()
            
            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key] if key != 'loss' else metrics['loss'].item()
            
            pbar.set_postfix({
                'loss': f"{metrics['loss'].item():.4f}",
                'slot_acc': f"{metrics['slot_acc']:.3f}",
                'temp': f"{temperature:.3f}"
            })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)
        
        epoch_metrics['epoch'] = epoch + 1
        epoch_metrics['temperature'] = temperature
        history.append(epoch_metrics)
        
        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Loss: {epoch_metrics['loss']:.4f}")
        print(f"  Slot Acc: {epoch_metrics['slot_acc']:.3f}")
        print(f"  Value Acc: {epoch_metrics['value_acc']:.3f}")
        print(f"  L_recon: {epoch_metrics['L_recon']:.4f}")
        print(f"  L_align: {epoch_metrics['L_align']:.4f}")
        print(f"  Temperature: {temperature:.3f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "sae_final.pt"
    torch.save({
        'epoch': start_epoch + args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'd_model': d_model,
    }, final_path)
    print(f"\nSaved final model to {final_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

if __name__ == "__main__":
    main()
