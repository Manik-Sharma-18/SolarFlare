"""
Solar Flare Prediction - Main Entry Point

This script loads configuration, prepares data, trains the model,
and generates visualizations.

Usage:
    python main.py                    # Use default config.yaml
    python main.py --config my.yaml   # Use custom config file
"""
import sys
import json
from pathlib import Path
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import SolarFluxPredictor
from solarflare_data import load_and_prepare_data, load_preprocessed_data
from solarflare_data.loader import create_dataloaders
from training import train_model, validate
from training.trainer import load_checkpoint
from utils import get_device, visualize_predictions
from utils.visualization import plot_training_history
from training.losses import get_loss_function


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_training(config: dict):
    """Main training pipeline."""
    
    # Setup device
    device = get_device(config['device']['use_cuda'])
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    use_preprocessed = config['data'].get('use_preprocessed', False)
    dual_channel = config['data'].get('dual_channel', False)
    
    if use_preprocessed:
        # Fast loading from preprocessed cubes
        train_dataset, val_dataset, test_dataset, metadata = load_preprocessed_data(
            preprocessed_dir=config['data']['preprocessed_dir'],
            t_in=config['data']['t_in'],
            t_out=config['data']['t_out'],
            train_split=config['data']['train_split'],
            val_split=config['data']['val_split'],
            augment_train=config['data']['augment'],
            dual_channel=dual_channel
        )
    else:
        # Load from raw structured arrays (slower)
        train_dataset, val_dataset, test_dataset, metadata = load_and_prepare_data(
            data_dir=config['data']['data_dir'],
            t_in=config['data']['t_in'],
            t_out=config['data']['t_out'],
            train_split=config['data']['train_split'],
            val_split=config['data']['val_split'],
            norm_method=config['normalization']['method'],
            norm_config=config['normalization'],
            augment_train=config['data']['augment'],
            dual_channel=dual_channel
        )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['training']['batch_size']
    )
    
    # Save metadata
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump(metadata, f, indent=2, default=str)
    
    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    output_channels = config['model'].get('output_channels', 1)
    use_checkpointing = config['model'].get('use_checkpointing', False)
    dropout_rate = config['model'].get('dropout_rate', 0.0)
    
    model = SolarFluxPredictor(
        input_channels=config['model']['input_channels'],
        output_channels=output_channels,
        t_out=config['data']['t_out'],
        channels=config['model']['channels'],
        kernel_size=config['model']['kernel_size'],
        downsample_input=config['model']['downsample_input'],
        use_checkpointing=use_checkpointing,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    # Print model info
    total_params = model.count_parameters()
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Input channels: {config['model']['input_channels']}")
    print(f"Output channels: {output_channels}")
    print(f"Channel progression: {config['model']['channels']}")
    print(f"Input downsampling: {config['model']['downsample_input']}")
    print(f"Gradient checkpointing: {use_checkpointing}")
    print(f"Dropout rate: {dropout_rate}")
    
    # Prepare training config
    train_config = {
        **config['training'],
        'save_dir': config['output']['save_dir'],
        'checkpoint_name': config['output']['checkpoint_name'],
        'save_history': config['output']['save_history'],
        'show_progress': config['logging']['progress_bar'],
        'loss': config.get('loss', {'type': 'l1'}),
        'output_channels': output_channels
    }
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = train_model(model, train_loader, val_loader, train_config, device)
    
    # Plot training history
    if config['output']['save_visualizations']:
        plot_training_history(history, str(output_dir / 'training_history.png'))
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)
    
    checkpoint_path = output_dir / config['output']['checkpoint_name']
    load_checkpoint(model, str(checkpoint_path), device)
    
    # Import loss function for validation
    loss_fn = get_loss_function(config.get('loss', {'type': 'l1'}))
    loss_fn = loss_fn.to(device)
    
    test_loss, test_mae_per_timestep = validate(
        model, test_loader, device,
        loss_fn=loss_fn,
        use_amp=config['training']['use_amp'],
        show_progress=config['logging']['progress_bar'],
        output_channels=output_channels
    )
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE per timestep: {test_mae_per_timestep}")
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_mae_per_timestep': test_mae_per_timestep.tolist()
    }
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Visualize predictions
    if config['output']['save_visualizations']:
        print("\n" + "=" * 60)
        print("VISUALIZING")
        print("=" * 60)
        
        visualize_predictions(
            model, test_dataset, device,
            n_samples=3,
            save_path=str(output_dir / 'predictions.png'),
            use_amp=config['training']['use_amp']
        )
    
    # Uncertainty Quantification (if enabled)
    uncertainty_config = config.get('uncertainty', {})
    if uncertainty_config.get('enabled', False):
        print("\n" + "=" * 60)
        print("UNCERTAINTY QUANTIFICATION")
        print("=" * 60)
        
        if model.dropout_rate == 0.0:
            print("Warning: Dropout rate is 0. Uncertainty estimation requires dropout_rate > 0.")
            print("Set model.dropout_rate in config.yaml and retrain for meaningful uncertainty.")
        else:
            from models.uncertainty import predict_with_uncertainty
            from utils.visualization import visualize_with_uncertainty
            
            n_samples = uncertainty_config.get('n_samples', 20)
            print(f"Running MC Dropout with {n_samples} samples...")
            
            # Evaluate uncertainty on a few test samples
            for i in range(min(3, len(test_dataset))):
                X_in, Y_out, _ = test_dataset[i]
                X_in = X_in.unsqueeze(0).to(device)
                Y_out = Y_out.unsqueeze(0).to(device)
                
                mean_pred, uncertainty = predict_with_uncertainty(
                    model, X_in, n_samples=n_samples
                )
                
                # Save uncertainty visualization
                if uncertainty_config.get('save_uncertainty_maps', True):
                    save_path = str(output_dir / f'uncertainty_sample_{i}.png')
                    visualize_with_uncertainty(
                        mean_pred, uncertainty, Y_out[:, :output_channels],
                        save_path=save_path
                    )
                
                # Print uncertainty statistics
                unc_mean = uncertainty.mean().item()
                unc_max = uncertainty.max().item()
                print(f"  Sample {i}: Mean uncertainty = {unc_mean:.6f}, Max = {unc_max:.6f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Outputs saved to: {output_dir}")


def run_inference(config: dict, checkpoint_path: str, data_path: str = None):
    """Run inference on new data using a trained model."""
    
    device = get_device(config['device']['use_cuda'])
    
    # Create model
    model = SolarFluxPredictor(
        input_channels=config['model']['input_channels'],
        output_channels=config['model'].get('output_channels', 1),
        t_out=config['data']['t_out'],
        channels=config['model']['channels'],
        kernel_size=config['model']['kernel_size'],
        downsample_input=config['model']['downsample_input'],
        use_checkpointing=config['model'].get('use_checkpointing', False),
        dropout_rate=config['model'].get('dropout_rate', 0.0)
    )
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    
    print("Model loaded and ready for inference")
    return model


# Example usage
if __name__ == '__main__':
    # Load config
    config_path = "config.yaml"
    if len(sys.argv) > 1 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Run training
    run_training(config)

