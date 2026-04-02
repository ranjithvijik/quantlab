"""
tests/conftest.py — top-level shared configuration.

Unit tests (tests/unit/) use a Streamlit stub via their own conftest.
Frontend tests (tests/frontend/) use the real Streamlit via AppTest.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests (no I/O)")
    config.addinivalue_line("markers", "integration: end-to-end tests")
    config.addinivalue_line("markers", "frontend: Streamlit UI tests")
    config.addinivalue_line("markers", "slow: slow tests")
