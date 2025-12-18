####################### CURRENT FINAL VERSION OF PROTOTYPE CONVLSTM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class SolarFluxDataset(Dataset):
    """Dataset for solar flux prediction with sliding windows"""

    def __init__(self, samples: List[Tuple], datasets: List[np.ndarray],
                 t_in: int = 20, t_out: int = 5, augment: bool = True):
        """
        Args:
            samples: List of (dataset_id, start_idx) tuples
            datasets: List of normalized flux cubes
            t_in: Input sequence length
            t_out: Output sequence length
            augment: Whether to apply augmentations
        """
        self.samples = samples
        self.datasets = datasets
        self.t_in = t_in
        self.t_out = t_out
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset_id, start_idx = self.samples[idx]
        data = self.datasets[dataset_id]

        # Extract sequence
        X_in = data[start_idx:start_idx + self.t_in]  # (T_in, H, W)
        Y_out = data[start_idx + self.t_in:start_idx + self.t_in + self.t_out]  # (T_out, H, W)

        # Apply sequence-level augmentations
        if self.augment:
            if np.random.rand() > 0.5:  # Horizontal flip
                X_in = np.flip(X_in, axis=2).copy()
                Y_out = np.flip(Y_out, axis=2).copy()
            if np.random.rand() > 0.5:  # Vertical flip
                X_in = np.flip(X_in, axis=1).copy()
                Y_out = np.flip(Y_out, axis=1).copy()

        # Convert to tensors: (C, T, H, W)
        X_in = torch.from_numpy(X_in).float().unsqueeze(0)  # (1, T_in, H, W)
        Y_out = torch.from_numpy(Y_out).float().unsqueeze(0)  # (1, T_out, H, W)

        return X_in, Y_out, (dataset_id, start_idx)


def load_and_prepare_data(data_dir: str, t_in: int = 20, t_out: int = 5,
                          norm_factor: float = 40000.0):
    """
    Load .npy files, create sliding windows, and split into train/val/test

    Args:
        data_dir: Path to directory containing .npy files
        t_in: Input sequence length
        t_out: Output sequence length
        norm_factor: Normalization factor

    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    data_path = Path(data_dir)

    # Check if directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_path}")

    # List all files for debugging
    all_files = list(data_path.iterdir())
    print(f"\nContents of {data_path}:")
    for f in all_files[:20]:  # Show first 20 files
        print(f"  {f.name}")
    if len(all_files) > 20:
        print(f"  ... and {len(all_files) - 20} more files")
    print()

    # Look for .npy files with windTotal pattern first
    npy_files = sorted(data_path.glob('windTotal*.npy'))

    # If no windTotal files, try any .npy files
    if len(npy_files) == 0:
        print("No windTotal*.npy files found, looking for any .npy files...")
        npy_files = sorted(data_path.glob('*.npy'))

    if len(npy_files) == 0:
        raise FileNotFoundError(
            f"No .npy files found in {data_path}\n"
            f"Expected files like: windTotal_MM_2024-05-08_0000_2348.npy"
        )

    print(f"Found {len(npy_files)} .npy files:")
    for f in npy_files:
        print(f"  {f.name}")
    print()

    datasets = []
    metadata = {
        'files': [],
        'shapes': [],
        'time_ranges': []
    }

    # Load each dataset
    for file_path in npy_files:
        print(f"Loading {file_path.name}...")
        try:
            data = np.load(file_path)

            # Extract unique coordinates and time
            x_coords = np.unique(data['X'])
            y_coords = np.unique(data['Y'])
            times = np.unique(data['time'])

            H, W = len(y_coords), len(x_coords)
            T = len(times)

            print(f"  Shape: T={T}, H={H}, W={W}")
            print(f"  Time range: {times[0]} to {times[-1]}")

            # Create spatial coordinate mapping
            x_to_idx = {x: i for i, x in enumerate(x_coords)}
            y_to_idx = {y: i for i, y in enumerate(y_coords)}
            time_to_idx = {t: i for i, t in enumerate(times)}

            # Reshape into dense cube (T, H, W)
            flux_cube = np.zeros((T, H, W), dtype=np.float32)

            # Fill cube efficiently
            for i in range(len(data)):
                t_idx = time_to_idx[data['time'][i]]
                h_idx = y_to_idx[data['Y'][i]]
                w_idx = x_to_idx[data['X'][i]]
                flux_cube[t_idx, h_idx, w_idx] = data['windTotal'][i]

            print(f"  Value range: [{flux_cube.min():.2f}, {flux_cube.max():.2f}]")
            print(f"  Mean: {flux_cube.mean():.2f}, Std: {flux_cube.std():.2f}")

            # Normalize
            flux_cube = flux_cube / norm_factor

            datasets.append(flux_cube)
            metadata['files'].append(file_path.name)
            metadata['shapes'].append((T, H, W))
            metadata['time_ranges'].append((str(times[0]), str(times[-1])))

        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(datasets) == 0:
        raise ValueError("No datasets were successfully loaded")

    # Create sliding window samples for each dataset
    all_samples_train = []
    all_samples_val = []
    all_samples_test = []

    for dataset_id, flux_cube in enumerate(datasets):
        T = flux_cube.shape[0]
        n_samples = T - t_in - t_out + 1

        if n_samples <= 0:
            print(f"Warning: Dataset {dataset_id} too short, skipping")
            continue

        # Split by time: 70% train, 15% val, 15% test
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)

        for start_idx in range(n_samples):
            sample = (dataset_id, start_idx)
            if start_idx < train_end:
                all_samples_train.append(sample)
            elif start_idx < val_end:
                all_samples_val.append(sample)
            else:
                all_samples_test.append(sample)

    print(f"\nDataset splits:")
    print(f"  Train: {len(all_samples_train)} samples")
    print(f"  Val:   {len(all_samples_val)} samples")
    print(f"  Test:  {len(all_samples_test)} samples")

    # Create datasets
    train_dataset = SolarFluxDataset(all_samples_train, datasets, t_in, t_out, augment=True)
    val_dataset = SolarFluxDataset(all_samples_val, datasets, t_in, t_out, augment=False)
    test_dataset = SolarFluxDataset(all_samples_test, datasets, t_in, t_out, augment=False)

    metadata['n_datasets'] = len(datasets)
    metadata['t_in'] = t_in
    metadata['t_out'] = t_out
    metadata['norm_factor'] = norm_factor

    return train_dataset, val_dataset, test_dataset, metadata


# ============================================================================
# ConvLSTM CELL & LAYER
# ============================================================================

class ConvLSTMCell(nn.Module):
    """ConvLSTM cell with forget gate bias initialization"""

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

        # Initialize forget gate bias to 1.0
        self._init_forget_bias()

    def _init_forget_bias(self):
        with torch.no_grad():
            # Gates are stacked as: i, f, g, o
            if self.conv.bias is not None:
                self.conv.bias[self.hidden_dim:2*self.hidden_dim].fill_(1.0)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    """ConvLSTM layer"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (B, C, T, H, W)
            hidden_state: list of (h, c) tuples
        Returns:
            outputs: (B, hidden_dim, T, H, W)
            last_state: list of (h, c) tuples
        """
        B, _, T, H, W = x.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(B, H, W, x.device)

        outputs = []

        for t in range(T):
            x_t = x[:, :, t]  # (B, C, H, W)

            for layer_idx, cell in enumerate(self.cell_list):
                h_prev, c_prev = hidden_state[layer_idx]
                h_next, c_next = cell(x_t, h_prev, c_prev)
                hidden_state[layer_idx] = (h_next, c_next)
                x_t = h_next

            outputs.append(h_next)

        outputs = torch.stack(outputs, dim=2)  # (B, hidden_dim, T, H, W)

        return outputs, hidden_state

    def _init_hidden(self, batch_size, height, width, device):
        return [(torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                 torch.zeros(batch_size, self.hidden_dim, height, width, device=device))
                for _ in range(self.num_layers)]


# ============================================================================
# MODEL ARCHITECTURE (Variant-B)
# ============================================================================

class SolarFluxPredictor(nn.Module):
    """ConvLSTM-based autoregressive solar flux predictor"""

    def __init__(self, input_channels=1, t_out=5):
        super().__init__()
        self.t_out = t_out

        # Preprocessing
        self.preprocess = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Encoder
        self.encoder_conv1 = ConvLSTM(32, 32, kernel_size=3)
        self.downsample1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_conv2 = ConvLSTM(64, 64, kernel_size=3)
        self.encoder_conv3 = ConvLSTM(64, 128, kernel_size=3)

        # Decoder input mapping
        self.decoder_input_conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)

        # Decoder downsampled path
        self.decoder_proj = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.decoder_conv2 = ConvLSTM(64, 64, kernel_size=3)
        self.decoder_conv3 = ConvLSTM(64, 128, kernel_size=3)

        # Decoder upsample and refinement
        self.upsample = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.refine_conv = ConvLSTM(64 + 32, 32, kernel_size=3)  # +32 for skip connection

        # Output head (residual)
        self.output_conv = nn.Conv2d(32, input_channels, kernel_size=1)

    def forward(self, x, teacher_forcing_ratio=0.0, y_true=None):
        """
        Args:
            x: (B, C, T_in, H, W)
            teacher_forcing_ratio: probability of using ground truth
            y_true: (B, C, T_out, H, W) ground truth for teacher forcing
        Returns:
            predictions: (B, C, T_out, H, W)
        """
        B, C, T_in, H, W = x.size()

        # Preprocess all input frames
        x_flat = x.view(B * T_in, C, H, W)
        x_prep = self.preprocess(x_flat)
        x_prep = x_prep.view(B, 32, T_in, H, W)

        # ENCODER
        # ConvLSTM1 at full resolution
        h1_seq, h1_states = self.encoder_conv1(x_prep)  # (B, 32, T_in, H, W)
        h1_last = h1_states[0][0]  # Save for skip connection (B, 32, H, W)

        # Downsample
        h1_down = self.downsample1(h1_seq[:, :, -1])  # (B, 64, H/2, W/2)
        h1_down = h1_down.unsqueeze(2).expand(-1, -1, T_in, -1, -1)

        # ConvLSTM2
        h2_seq, h2_states = self.encoder_conv2(h1_down)

        # ConvLSTM3 (latent)
        h3_seq, h3_states = self.encoder_conv3(h2_seq)
        h_latent = h3_states[0][0]  # (B, 128, H/2, W/2)

        # DECODER (Autoregressive)
        predictions = []
        input_frame = x[:, :, -1]  # Last frame of input (B, C, H, W)

        # Initialize decoder states from encoder
        decoder_h2 = h2_states[0][0]
        decoder_c2 = h2_states[0][1]
        decoder_h3 = h3_states[0][0]
        decoder_c3 = h3_states[0][1]
        decoder_state2 = [(decoder_h2, decoder_c2)]
        decoder_state3 = [(decoder_h3, decoder_c3)]

        # Initialize refine state
        refine_state = None

        for t in range(self.t_out):
            # Map input frame to decoder input
            dec_input = self.decoder_input_conv(input_frame)  # (B, 32, H, W)

            # Downsample
            dec_down = self.decoder_proj(dec_input)  # (B, 64, H/2, W/2)
            dec_down = dec_down.unsqueeze(2)  # (B, 64, 1, H/2, W/2)

            # Run decoder ConvLSTM stack
            dec_h2, decoder_state2 = self.decoder_conv2(dec_down, decoder_state2)
            dec_h3, decoder_state3 = self.decoder_conv3(dec_h2, decoder_state3)

            # Upsample
            dec_up = self.upsample(dec_h3[:, :, 0])  # (B, 64, H, W)

            # Concatenate with skip connection
            dec_concat = torch.cat([dec_up, h1_last], dim=1)  # (B, 96, H, W)
            dec_concat = dec_concat.unsqueeze(2)  # (B, 96, 1, H, W)

            # Refine
            refined, refine_state = self.refine_conv(dec_concat, refine_state)

            # Output residual
            delta = self.output_conv(refined[:, :, 0])  # (B, C, H, W)

            # Predict frame
            pred_frame = input_frame + delta
            predictions.append(pred_frame)

            # Teacher forcing
            if teacher_forcing_ratio > 0 and y_true is not None and np.random.rand() < teacher_forcing_ratio:
                input_frame = y_true[:, :, t]
            else:
                input_frame = pred_frame

        predictions = torch.stack(predictions, dim=2)  # (B, C, T_out, H, W)

        return predictions


# ============================================================================
# TRAINING
# ============================================================================

def compute_metrics(pred, target):
    """Compute MAE and SSIM"""
    mae = F.l1_loss(pred, target, reduction='none')
    mae_per_timestep = mae.mean(dim=(0, 1, 3, 4))  # Average over B, C, H, W
    mae_total = mae.mean()

    return {
        'mae_total': mae_total.item(),
        'mae_per_timestep': mae_per_timestep.cpu().numpy()
    }


def train_epoch(model, dataloader, optimizer, scaler, device, teacher_forcing_ratio, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for X_in, Y_out, _ in pbar:
        X_in = X_in.to(device)
        Y_out = Y_out.to(device)

        optimizer.zero_grad()

        with autocast():
            predictions = model(X_in, teacher_forcing_ratio, Y_out)
            loss = F.l1_loss(predictions, Y_out)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_mae_per_timestep = []

    with torch.no_grad():
        for X_in, Y_out, _ in tqdm(dataloader, desc="Validating"):
            X_in = X_in.to(device)
            Y_out = Y_out.to(device)

            with autocast():
                predictions = model(X_in, teacher_forcing_ratio=0.0)
                loss = F.l1_loss(predictions, Y_out)

            total_loss += loss.item()
            metrics = compute_metrics(predictions, Y_out)
            all_mae_per_timestep.append(metrics['mae_per_timestep'])

    avg_loss = total_loss / len(dataloader)
    avg_mae_per_timestep = np.mean(all_mae_per_timestep, axis=0)

    return avg_loss, avg_mae_per_timestep


def train_model(model, train_loader, val_loader, config, device):
    """Main training loop"""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )

    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae_per_timestep': []}

    for epoch in range(1, config['epochs'] + 1):
        # Teacher forcing schedule: linear decay
        tf_ratio = max(0.0, config['tf_start'] - (config['tf_start'] / config['epochs']) * epoch)

        print(f"\nEpoch {epoch}/{config['epochs']} - Teacher Forcing: {tf_ratio:.3f}")

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, tf_ratio, epoch)
        val_loss, val_mae_per_timestep = validate(model, val_loader, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val MAE per timestep: {val_mae_per_timestep}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_per_timestep'].append(val_mae_per_timestep.tolist())

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'best_model.pt')
            print("✓ Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch}")
            break

    return history


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, dataset, device, n_samples=3, save_path='predictions.png'):
    """Visualize predictions vs ground truth"""
    model.eval()

    fig, axes = plt.subplots(n_samples, 7, figsize=(21, 3*n_samples))

    with torch.no_grad():
        for i in range(n_samples):
            X_in, Y_out, (ds_id, start_idx) = dataset[i * len(dataset) // n_samples]
            X_in = X_in.unsqueeze(0).to(device)
            Y_out = Y_out.unsqueeze(0).to(device)

            with autocast():
                predictions = model(X_in, teacher_forcing_ratio=0.0)

            # Plot last input frame
            axes[i, 0].imshow(X_in[0, 0, -1].cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
            axes[i, 0].set_title(f'Input t={dataset.t_in}')
            axes[i, 0].axis('off')

            # Plot predictions and ground truth
            for t in range(dataset.t_out):
                axes[i, 1+t].imshow(predictions[0, 0, t].cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
                axes[i, 1+t].set_title(f'Pred t+{t+1}')
                axes[i, 1+t].axis('off')

            axes[i, 6].imshow(Y_out[0, 0, -1].cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
            axes[i, 6].set_title(f'GT t+{dataset.t_out}')
            axes[i, 6].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Configuration
    config = {
        'data_dir': r'C:\Users\Manik\Solar model Thesis\winding_data',
        't_in': 10,
        't_out': 3,
        'batch_size': 1,
        'epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'tf_start': 0.9,  # Teacher forcing start probability
        'patience': 10
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    train_dataset, val_dataset, test_dataset, metadata = load_and_prepare_data(
        config['data_dir'],
        config['t_in'],
        config['t_out']
    )

    # Save metadata
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=True)

    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = SolarFluxPredictor(input_channels=1, t_out=config['t_out'])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    history = train_model(model, train_loader, val_loader, config, device)

    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Load best model and evaluate on test set
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_mae_per_timestep = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE per timestep: {test_mae_per_timestep}")

    # Visualize predictions
    print("\n" + "="*60)
    print("VISUALIZING")
    print("="*60)
    visualize_predictions(model, test_dataset, device, n_samples=3)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()