#!/usr/bin/env python3
"""
Test that token validation catches all permutations of missing model licenses.
"""

import sys
import os

# Setup path: this script lives in tests/manual/, so the repo root is two up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock huggingface_hub BEFORE importing diarization
from unittest.mock import MagicMock, patch
import huggingface_hub
from huggingface_hub.utils import GatedRepoError

# Derived from the same constant validate_hf_token checks against, so this
# script cannot drift out of date the way the old hardcoded list did (it
# still listed two gated repos after a third was added).
from src.core.diarization import GATED_REPOS

MODELS = sorted(GATED_REPOS)


def run_validation_with_mocked_access(accessible_repos):
    """Run validation with mocked repo access."""

    def mock_list_repo_files(repo_id, token=None):
        if repo_id in accessible_repos:
            return ["config.yaml", "model.bin"]
        else:
            raise GatedRepoError(f"Access denied to gated repo {repo_id}")

    mock_api_instance = MagicMock()
    mock_api_instance.whoami.return_value = {"name": "testuser"}

    with patch.object(huggingface_hub, 'HfApi', return_value=mock_api_instance):
        with patch.object(huggingface_hub, 'list_repo_files', side_effect=mock_list_repo_files):
            # Import fresh each time to use mocks
            import importlib
            import src.core.diarization as diarization
            importlib.reload(diarization)

            return diarization.validate_hf_token("hf_test_token_12345")


def test_all_permutations():
    """Test every permutation of gated-model access (2^N for N gated models)."""

    print("=" * 70)
    print("Testing Token Validation - All Model Access Permutations")
    print("=" * 70)
    print()
    print(f"Required models:")
    for m in MODELS:
        print(f"  - {m}")
    print()

    all_passed = True

    # Generate every permutation (one bit per gated model)
    for i in range(2 ** len(MODELS)):
        accessible = []
        missing = []
        for j, model in enumerate(MODELS):
            if (i >> j) & 1:
                accessible.append(model)
            else:
                missing.append(model)

        # Create description
        desc_parts = []
        for model in MODELS:
            short = model.split("/")[1][:15]
            if model in accessible:
                desc_parts.append(f"+{short}")
            else:
                desc_parts.append(f"-{short}")
        desc = " | ".join(desc_parts)

        # Run validation
        is_valid, message = run_validation_with_mocked_access(accessible)

        # Check result - all models accessible means valid
        all_accessible = len(accessible) == len(MODELS)

        if all_accessible:
            # Should be valid
            if is_valid:
                print(f"[PASS] {desc}")
                print(f"       Valid: {message[:60]}")
            else:
                print(f"[FAIL] {desc}")
                print(f"       Should be VALID but got: {message[:60]}")
                all_passed = False
        else:
            # Should be invalid AND mention ALL missing models
            if is_valid:
                print(f"[FAIL] {desc}")
                print(f"       Should be INVALID but was valid")
                all_passed = False
            else:
                # Check that all missing models are mentioned
                missing_mentioned = []
                missing_not_mentioned = []
                for m in missing:
                    short_name = m.split("/")[1]
                    if short_name in message:
                        missing_mentioned.append(short_name)
                    else:
                        missing_not_mentioned.append(short_name)

                if missing_not_mentioned:
                    print(f"[FAIL] {desc}")
                    print(f"       Missing models NOT in error: {missing_not_mentioned}")
                    print(f"       Error was: {message[:80]}")
                    all_passed = False
                else:
                    print(f"[PASS] {desc}")
                    print(f"       Correctly lists: {missing_mentioned}")
        print()

    return all_passed


def test_invalid_token_formats():
    """Test that invalid token formats are rejected."""
    print("=" * 70)
    print("Testing Invalid Token Formats")
    print("=" * 70 + "\n")

    # These don't need mocking - they fail before API calls
    from src.core.diarization import validate_hf_token

    test_cases = [
        ("", "empty token"),
        ("   ", "whitespace only"),
        ("not_a_token", "doesn't start with hf_"),
        ("hf_", "just prefix, no content"),
    ]

    all_passed = True
    for token, desc in test_cases:
        is_valid, message = validate_hf_token(token)
        if is_valid:
            print(f"[FAIL] {desc}: should be invalid but was valid")
            all_passed = False
        else:
            print(f"[PASS] {desc}: correctly rejected - {message[:40]}")

    print()
    return all_passed


if __name__ == "__main__":
    print("\n")

    # Test invalid formats first (no mocking needed)
    format_ok = test_invalid_token_formats()

    # Test all access permutations
    perm_ok = test_all_permutations()

    print("=" * 70)
    if perm_ok and format_ok:
        print("=== ALL TESTS PASSED ===")
        sys.exit(0)
    else:
        print("=== SOME TESTS FAILED ===")
        sys.exit(1)
