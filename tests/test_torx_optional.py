"""Core stays importable when the torx extra is absent.

These tests are not marked ``torx``. They do not import ``torx``.
"""

from __future__ import annotations

import builtins
import sys

import pytest

import unsga3_extropic
import unsga3_extropic.torx_circuit as torx_circuit
from unsga3_extropic.backends.base import SamplingBackend
from unsga3_extropic.torx_circuit import TorxPswapCircuit


def test_core_package_imports():
    assert unsga3_extropic.__version__


def test_circuit_module_does_not_bind_torx():
    assert "torx" not in torx_circuit.__dict__
    assert "jax" not in torx_circuit.__dict__
    assert getattr(torx_circuit, "DiscretePCircuit", None) is None
    assert not isinstance(TorxPswapCircuit(), SamplingBackend)


def test_python_310_refuses_the_extra(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 10, 12, "final", 0))
    with pytest.raises(ImportError, match="Python >= 3.11"):
        TorxPswapCircuit().sample(n_samples=1)


def test_missing_extra_raises_import_error(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torx" or name.startswith("torx."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ImportError, match="extro-torx"):
        TorxPswapCircuit().sample(n_samples=1)
