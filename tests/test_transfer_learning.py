"""Tests for transfer learning: partial loading, freezing, param groups."""
import pytest
import torch
import tempfile
from pathlib import Path


CHANNELS = [32, 64, 128]
KERNEL_SIZE = 5
H, W = 16, 16
T_IN = 3
T_OUT = 2


def _make_model(input_channels, **kwargs):
    """Create a SolarFluxPredictor with given input_channels."""
    from models.predictor import SolarFluxPredictor
    defaults = dict(
        output_channels=1, t_out=T_OUT, channels=CHANNELS,
        kernel_size=KERNEL_SIZE,
        use_sa_convlstm=True, temporal_attention=True,
        attention_gate=True, delta_scale_init=0.5, dropout_rate=0.0,
    )
    defaults.update(kwargs)
    return SolarFluxPredictor(input_channels=input_channels, **defaults)


def _save_checkpoint(model, path):
    """Save a minimal checkpoint matching the project format."""
    torch.save({
        'epoch': 5,
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': {},
        'scheduler_state_dict': None,
        'best_val_loss': 1.0,
        'patience_counter': 0,
        'normalization_params': {'method': 'asinh', 'scale': 1.0},
        'config': {},
        'checkpoint_version': 1,
    }, path)


class TestPartialCheckpointLoading:
    """Tests for load_pretrained_weights with mismatched input channels."""

    def test_load_mismatched_input_channels(self, tmp_path):
        """C=1 checkpoint loads into C=2 model; matching keys loaded."""
        from utils.transfer import load_pretrained_weights

        pretrain_model = _make_model(input_channels=1)
        ckpt_path = tmp_path / "pretrained.pt"
        _save_checkpoint(pretrain_model, ckpt_path)

        finetune_model = _make_model(input_channels=2)
        loaded, skipped, reinited = load_pretrained_weights(
            finetune_model, str(ckpt_path), reinit_mismatched=True,
        )

        assert len(loaded) > 0, "Should load some keys"
        assert len(skipped) > 0, "Should skip mismatched keys"
        # Only preprocess.0.weight and decoder_input_conv.weight should mismatch
        mismatched_weights = [k for k in skipped if 'weight' in k]
        assert 'preprocess.0.weight' in mismatched_weights
        assert 'decoder_input_conv.weight' in mismatched_weights

    def test_matching_keys_are_identical(self, tmp_path):
        """Encoder weights match exactly after partial load."""
        from utils.transfer import load_pretrained_weights

        pretrain_model = _make_model(input_channels=1)
        pretrain_state = {k: v.clone() for k, v in pretrain_model.state_dict().items()}
        ckpt_path = tmp_path / "pretrained.pt"
        _save_checkpoint(pretrain_model, ckpt_path)

        finetune_model = _make_model(input_channels=2)
        loaded, _, _ = load_pretrained_weights(finetune_model, str(ckpt_path))

        for key in loaded:
            assert torch.equal(
                finetune_model.state_dict()[key], pretrain_state[key]
            ), f"Key '{key}' should match pretrained weights"

    def test_mismatched_keys_are_reinitialized(self, tmp_path):
        """Input layers are reinitialized, not zero or from checkpoint."""
        from utils.transfer import load_pretrained_weights

        pretrain_model = _make_model(input_channels=1)
        ckpt_path = tmp_path / "pretrained.pt"
        _save_checkpoint(pretrain_model, ckpt_path)

        finetune_model = _make_model(input_channels=2)
        _, _, reinited = load_pretrained_weights(finetune_model, str(ckpt_path))

        assert len(reinited) > 0
        # Reinitialized weights should not be all zeros
        for key in reinited:
            param = dict(finetune_model.named_parameters())[key]
            if param.dim() >= 2:
                assert not torch.allclose(param, torch.zeros_like(param)), (
                    f"Reinitialized '{key}' should not be all zeros"
                )

    def test_partial_load_produces_finite_output(self, tmp_path):
        """Model produces finite output after partial checkpoint load."""
        from utils.transfer import load_pretrained_weights

        pretrain_model = _make_model(input_channels=1)
        ckpt_path = tmp_path / "pretrained.pt"
        _save_checkpoint(pretrain_model, ckpt_path)

        finetune_model = _make_model(input_channels=2)
        load_pretrained_weights(finetune_model, str(ckpt_path))
        finetune_model.eval()

        x = torch.randn(1, 2, T_IN, H, W)
        with torch.no_grad():
            out = finetune_model(x)
        assert torch.isfinite(out).all(), "Output should be finite after partial load"
        assert out.shape == (1, 1, T_OUT, H, W)

    def test_same_channels_loads_everything(self, tmp_path):
        """When channels match, all keys are loaded, none skipped."""
        from utils.transfer import load_pretrained_weights

        model_a = _make_model(input_channels=1)
        ckpt_path = tmp_path / "pretrained.pt"
        _save_checkpoint(model_a, ckpt_path)

        model_b = _make_model(input_channels=1)
        loaded, skipped, reinited = load_pretrained_weights(model_b, str(ckpt_path))

        assert len(skipped) == 0, "No keys should be skipped when channels match"
        assert len(reinited) == 0
        assert len(loaded) == len(model_a.state_dict())


class TestLayerFreezing:
    """Tests for freeze_encoder and unfreeze_all."""

    def test_freeze_encoder_parameters(self):
        """Frozen encoder params have requires_grad=False."""
        from utils.transfer import freeze_encoder

        model = _make_model(input_channels=2)
        frozen = freeze_encoder(model)

        assert len(frozen) > 0
        for name, param in model.named_parameters():
            if name in frozen:
                assert not param.requires_grad, f"'{name}' should be frozen"

    def test_decoder_stays_unfrozen(self):
        """Decoder, attention, output layers remain trainable after freeze."""
        from utils.transfer import freeze_encoder

        model = _make_model(input_channels=2)
        freeze_encoder(model)

        unfrozen_prefixes = (
            'decoder_', 'upsample.', 'refine_conv.', 'output_conv.',
            'temporal_attn.', 'attn_gate.', 'delta_scale', 'preprocess.',
        )
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in unfrozen_prefixes):
                assert param.requires_grad, f"'{name}' should remain unfrozen"

    def test_unfreeze_restores_grad(self):
        """After unfreezing, all params have requires_grad=True."""
        from utils.transfer import freeze_encoder, unfreeze_all

        model = _make_model(input_channels=2)
        freeze_encoder(model)
        unfreeze_all(model)

        for name, param in model.named_parameters():
            assert param.requires_grad, f"'{name}' should be unfrozen"

    def test_frozen_model_forward_works(self):
        """Forward pass works with frozen encoder."""
        from utils.transfer import freeze_encoder

        model = _make_model(input_channels=2)
        freeze_encoder(model)
        model.eval()

        x = torch.randn(1, 2, T_IN, H, W)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()
        assert out.shape == (1, 1, T_OUT, H, W)


class TestFinetuneParamGroups:
    """Tests for differential learning rate param groups."""

    def test_two_groups_returned(self):
        """get_finetune_param_groups returns 2 groups with different LRs."""
        from utils.transfer import get_finetune_param_groups

        model = _make_model(input_channels=2)
        groups = get_finetune_param_groups(model, base_lr=0.001, lr_scale_pretrained=0.1)

        assert len(groups) == 2
        assert groups[0]['lr'] == 0.001       # new/input layers
        assert groups[1]['lr'] == 0.0001      # pretrained layers

    def test_frozen_params_excluded(self):
        """Frozen params are not included in any param group."""
        from utils.transfer import freeze_encoder, get_finetune_param_groups

        model = _make_model(input_channels=2)
        freeze_encoder(model)
        groups = get_finetune_param_groups(model, base_lr=0.001)

        all_group_params = set()
        for g in groups:
            for p in g['params']:
                all_group_params.add(id(p))

        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert id(param) not in all_group_params, (
                    f"Frozen param '{name}' should not be in optimizer groups"
                )

    def test_all_trainable_params_covered(self):
        """All trainable params appear in exactly one group."""
        from utils.transfer import get_finetune_param_groups

        model = _make_model(input_channels=2)
        groups = get_finetune_param_groups(model, base_lr=0.001)

        group_param_ids = set()
        for g in groups:
            for p in g['params']:
                group_param_ids.add(id(p))

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert id(param) in group_param_ids, (
                    f"Trainable param '{name}' should be in a group"
                )


class TestIsInputDependent:
    """Tests for input-dependent layer identification."""

    def test_preprocess_is_input_dependent(self):
        from utils.transfer import is_input_dependent
        assert is_input_dependent('preprocess.0.weight')
        assert is_input_dependent('preprocess.0.bias')

    def test_decoder_input_conv_is_input_dependent(self):
        from utils.transfer import is_input_dependent
        assert is_input_dependent('decoder_input_conv.weight')
        assert is_input_dependent('decoder_input_conv.bias')

    def test_encoder_is_not_input_dependent(self):
        from utils.transfer import is_input_dependent
        assert not is_input_dependent('encoder_conv1.cell_list.0.convlstm_cell.conv.weight')
        assert not is_input_dependent('downsample1.weight')

    def test_decoder_conv_is_not_input_dependent(self):
        from utils.transfer import is_input_dependent
        assert not is_input_dependent('decoder_conv2.cell_list.0.convlstm_cell.conv.weight')
        assert not is_input_dependent('output_conv.weight')
