
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.models import GPT, Block, MLP, CausalSelfAttention
from types import SimpleNamespace
from src.utils import init_from_checkpoint


@pytest.fixture
def config():
    """Create a config object for testing"""
    return SimpleNamespace(
        wandb_project_name="gpt2_small",
        wandb_experiment_name="gpt2-dev",
        enable_wandb_logging=False,
        logs_dir="logs",
        checkpoint_dir="checkpoints",
        device_type="cuda",
        sequence_length=1024,
        vocab_size=50257,
        embedding_dim=768,
        n_layer=12,
        n_head=12,
        total_batch_size=524288,
        micro_batch_size=16,
        max_lr=6e-4,
        min_lr=6e-5,
        warmup_iters=715,
        max_iters=6,
        evaluation_interval=2,
        checkpoint_interval=2,
    )



# Test for init_from_checkpoint
def test_init_from_checkpoint(config, tmp_path):
    """Test loading a model and iter_num from a checkpoint"""

    # Create a mock model and save its state_dict with additional metadata
    model = GPT(config)
    iter_num = 42  # Example iteration number
    mock_checkpoint = {
        "model": model.state_dict(),
        "iter_num": iter_num
    }

    # Save the mock checkpoint
    ckpt_path = tmp_path / "mock_checkpoint.pth"
    torch.save(mock_checkpoint, ckpt_path)

    # Load the checkpoint into a new model
    new_model = GPT(config)
    loaded_model, loaded_iter_num = init_from_checkpoint(new_model, str(ckpt_path), config.device_type)

    # Verify the loaded state_dict matches the original
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2), "Loaded model parameters do not match saved parameters"

    # Verify the loaded iter_num
    assert loaded_iter_num == iter_num, f"Expected iter_num {iter_num}, but got {loaded_iter_num}"