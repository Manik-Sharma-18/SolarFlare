"""Tests for checkpoint save/load roundtrip, atomic saves, and error handling (TST-03)."""
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_model(channels=None):
    """Create a tiny SolarFluxPredictor for testing."""
    from models.predictor import SolarFluxPredictor

    if channels is None:
        channels = [4, 8, 16]
    return SolarFluxPredictor(
        input_channels=1,
        output_channels=1,
        t_out=2,
        channels=channels,
        kernel_size=3,
        downsample_input=True,
        use_checkpointing=False,
        dropout_rate=0.0,
    )


def _make_dummy_scaler():
    """Return a DummyGradScaler instance."""
    from utils.device import _DummyGradScaler

    return _DummyGradScaler()


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    """Verify checkpoint save -> load produces identical state."""

    def test_checkpoint_roundtrip_identical_output(self, tmp_path):
        from utils.checkpoint import save_checkpoint, load_checkpoint_for_resume

        model = _make_tiny_model()
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = _make_dummy_scaler()

        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            output_before = model(x).clone()

        filepath = tmp_path / "ckpt.pt"
        save_checkpoint(
            filepath,
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            best_val_loss=0.5,
            patience_counter=2,
            normalization_params={"method": "asinh", "scale": 1.0},
            config={"device": "cpu"},
            history={"train_loss": [1.0, 0.5]},
        )

        model2 = _make_tiny_model()
        model2.eval()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        scaler2 = _make_dummy_scaler()

        load_checkpoint_for_resume(
            filepath,
            model2,
            optimizer2,
            scheduler=None,
            scaler=scaler2,
            device=torch.device("cpu"),
        )

        with torch.no_grad():
            output_after = model2(x)

        assert torch.allclose(output_before, output_after, atol=1e-6), (
            f"Max diff: {(output_before - output_after).abs().max().item()}"
        )

    def test_checkpoint_roundtrip_state_restored(self, tmp_path):
        from utils.checkpoint import save_checkpoint, load_checkpoint_for_resume

        model = _make_tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = _make_dummy_scaler()

        filepath = tmp_path / "ckpt.pt"
        save_checkpoint(
            filepath,
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            best_val_loss=0.123,
            patience_counter=3,
            normalization_params={"method": "asinh"},
            config={"device": "cpu"},
            history={},
        )

        model2 = _make_tiny_model()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        scaler2 = _make_dummy_scaler()

        start_epoch, best_val_loss, patience_counter, history, norm_params = (
            load_checkpoint_for_resume(
                filepath,
                model2,
                optimizer2,
                scheduler=None,
                scaler=scaler2,
                device=torch.device("cpu"),
            )
        )

        assert start_epoch == 6, f"Expected epoch 6, got {start_epoch}"
        assert best_val_loss == pytest.approx(0.123)
        assert patience_counter == 3

    def test_checkpoint_roundtrip_normalization_params(self, tmp_path):
        from utils.checkpoint import save_checkpoint, load_checkpoint

        model = _make_tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = _make_dummy_scaler()

        norm_params = {"method": "asinh", "asinh_softening": 0.005, "scale": 3.0}
        filepath = tmp_path / "ckpt.pt"
        save_checkpoint(
            filepath,
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            best_val_loss=1.0,
            patience_counter=0,
            normalization_params=norm_params,
            config={"device": "cpu"},
            history={},
        )

        checkpoint = load_checkpoint(filepath)
        loaded_norm = checkpoint["normalization_params"]

        assert loaded_norm == norm_params, f"Expected {norm_params}, got {loaded_norm}"


# ---------------------------------------------------------------------------
# Atomic save tests
# ---------------------------------------------------------------------------


class TestAtomicSave:
    """Verify _atomic_save creates files correctly with no leftovers."""

    def test_atomic_save_creates_file(self, tmp_path):
        from utils.checkpoint import _atomic_save

        filepath = tmp_path / "test.pt"
        _atomic_save({"test": 1}, filepath)

        assert filepath.exists()
        loaded = torch.load(filepath, map_location="cpu", weights_only=False)
        assert loaded == {"test": 1}

    def test_atomic_save_no_temp_files_left(self, tmp_path):
        from utils.checkpoint import _atomic_save

        filepath = tmp_path / "test.pt"
        _atomic_save({"test": 1}, filepath)

        temp_files = list(tmp_path.glob(".tmp_ckpt_*"))
        assert len(temp_files) == 0, f"Temp files remain: {temp_files}"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestCheckpointErrors:
    """Verify correct errors for missing files, version mismatches, and arch mismatches."""

    def test_load_checkpoint_missing_file(self):
        from utils.checkpoint import load_checkpoint

        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.pt")

    def test_load_checkpoint_version_mismatch(self, tmp_path):
        from utils.checkpoint import load_checkpoint

        filepath = tmp_path / "bad_version.pt"
        torch.save({"checkpoint_version": 999}, filepath)

        with pytest.raises(RuntimeError, match="version mismatch"):
            load_checkpoint(filepath)

    def test_checkpoint_architecture_mismatch(self, tmp_path):
        from models.predictor import SolarFluxPredictor
        from utils.checkpoint import save_checkpoint, load_checkpoint_for_resume

        # Save from model with channels=[4, 8, 16]
        model_a = _make_tiny_model(channels=[4, 8, 16])
        optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
        scaler = _make_dummy_scaler()

        filepath = tmp_path / "ckpt.pt"
        save_checkpoint(
            filepath,
            epoch=1,
            model=model_a,
            optimizer=optimizer_a,
            scheduler=None,
            scaler=scaler,
            best_val_loss=1.0,
            patience_counter=0,
            normalization_params={},
            config={"device": "cpu"},
            history={},
        )

        # Try loading into model with channels=[8, 16, 32]
        model_b = _make_tiny_model(channels=[8, 16, 32])
        optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
        scaler_b = _make_dummy_scaler()

        with pytest.raises(RuntimeError, match="size mismatch"):
            load_checkpoint_for_resume(
                filepath,
                model_b,
                optimizer_b,
                scheduler=None,
                scaler=scaler_b,
                device=torch.device("cpu"),
            )


# ---------------------------------------------------------------------------
# Config diff tests
# ---------------------------------------------------------------------------


class TestConfigDiff:
    """Verify _diff_configs detects changes and returns empty for identical configs."""

    def test_config_diff_detects_changes(self):
        from utils.checkpoint import _diff_configs

        diffs = _diff_configs({"lr": 0.001}, {"lr": 0.0001})
        assert len(diffs) == 1
        assert "lr" in diffs[0]
        assert "0.001" in diffs[0]
        assert "0.0001" in diffs[0]

    def test_config_diff_empty_for_identical(self):
        from utils.checkpoint import _diff_configs

        diffs = _diff_configs({"a": 1, "b": "hello"}, {"a": 1, "b": "hello"})
        assert diffs == []
