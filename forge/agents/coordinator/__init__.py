"""Coordinator Agent (Agent 3) - The brain of Inference Forge."""

from .agent import CoordinatorAgent
from .synthesis import ExperimentSynthesizer, SynthesisConfig

__all__ = ["CoordinatorAgent", "ExperimentSynthesizer", "SynthesisConfig"]
