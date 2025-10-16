"""Placeholder test module.

Ensures pytest finds a test module under the configured `testpaths` so it does
not fall back to recursive discovery outside `tests/`.

The single test below is skipped intentionally. This makes the CI step succeed
even when there are no real tests yet, avoiding a nonzero exit due to "no tests
collected".
"""

import pytest


@pytest.mark.skip(reason="placeholder – no tests yet")
def test_placeholder() -> None:
    pass
