"""
ACMEModels Repository
Introductory remarks: This module is part of the ACMEModels codebase.

Central configuration constants for the ACME Models CLI.
"""

from __future__ import annotations

# LLM configuration ---------------------------------------------------------

LLM_ANALYSIS_MODEL = "llama3.1:latest"
"""Model identifier used for reasoning-oriented prompts."""

LLM_EXTRACTION_MODEL = LLM_ANALYSIS_MODEL
"""Model identifier used for short extraction prompts."""

LLM_TEMPERATURE = 0.0
"""Default temperature for deterministic responses."""
