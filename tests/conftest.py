"""Shared fixtures for the Von Mises neuron test suite."""

import torch
import pytest


@pytest.fixture
def device():
    """Auto-detect CUDA device for tests, falling back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
