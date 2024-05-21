import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.models import GPT, Block, MLP, CausalSelfAttention
from types import SimpleNamespace


@pytest.fixture
def config():
    """Create a config object for testing"""
    return SimpleNamespace(
        wandb_project_name="gpt2_small",
        wandb_experiment_name="gpt2-dev",
        enable_wandb_logging=True,
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


# Input output shape testing
def test_causal_self_attention(config):
    """Test forward pass for self-attention"""
    batch_size, seq_length = 16, config.sequence_length
    model = CausalSelfAttention(config)
    x = torch.rand(batch_size, seq_length, config.embedding_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_length, config.embedding_dim), "Output shape mismatch for self-attention"


def test_mlp(config):
    """Test forward pass for MLP"""
    batch_size, seq_length = 16, config.sequence_length
    model = MLP(config)
    x = torch.rand(batch_size, seq_length, config.embedding_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_length, config.embedding_dim), "Output shape mismatch for MLP"


def test_block(config):
    """Test forward pass of a single transformer Block"""
    batch_size, seq_length = 16, config.sequence_length
    model = Block(config)
    x = torch.rand(batch_size, seq_length, config.embedding_dim)
    output = model(x)
    assert output.shape == (batch_size, seq_length, config.embedding_dim), "Output shape mismatch for transformer block"


def test_gpt_forward(config):
    """Test forward pass of the full GPT model"""
    batch_size, seq_length = 16, config.sequence_length
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    logits, _ = model(idx)
    assert logits.shape == (batch_size, seq_length, config.vocab_size), "output shape mismatch for GPT logits"


# Loss values testing 
def test_gpt_with_loss(config):
    """Test GPT forward pass with targets and loss"""
    batch_size, seq_length = 16, config.sequence_length
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (batch_size, seq_length, config.vocab_size), "Output shape mismatch for GPT logits"
    assert loss is not None, "loss should not be a None value when targets are supplied"
    assert loss.item() > 0, "loss value should be positive"
# add testing for expected initial loss according to number of classes 


def test_optimizer_configuration(config):
    """Test GPT optimizer """
    model = GPT(config)
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=config.max_lr, device_type="cpu")
    assert isinstance(optimizer, torch.optim.AdamW), "Optimizer should be AdamW "
    assert len(optimizer.param_groups) == 2, "Optimizer should have two parameter groups: decay and no-decay"




def test_gradient_flow(config):
    """Test gradient flow through the model"""
    batch_size, seq_length = 16, config.sequence_length
    model = GPT(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    logits, loss = model(idx, targets=targets)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(gradients) > 0, "Gradients should be non-zero after backward pass"
    for grad in gradients:
        assert grad.norm().item() > 0, "Gradient norm should be positive"
    # Think of more tests



def test_pretrained_loading(config):
    """Test loading a pretrained GPT model"""
    model = GPT.from_pretrained("gpt2")
    assert isinstance(model, GPT), "Model should be an instance of GPT after loading"

def test_model_save_and_load(config, tmp_path):
    """Test saving and loading the model"""
    model = GPT(config)
    save_path = tmp_path / "gpt_model.pth"
    torch.save(model.state_dict(), save_path)
    loaded_model = GPT(config)
    loaded_model.load_state_dict(torch.load(save_path))
    # Ensure the loaded model's parameters are identical
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2), "Saved and loaded model parameters do not match"
