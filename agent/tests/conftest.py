"""
Shared pytest fixtures for agent unit tests.
Sets a dummy OPENAI_API_KEY so UniversalAgent can be instantiated
without a real key (tests that actually call the LLM are skipped or mocked).
"""
import os
import pytest

# Set before any imports so OpenAI client doesn't raise on init
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key-for-unit-tests")
