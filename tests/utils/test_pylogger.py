"""Tests for RankedLogger."""

import logging
from unittest.mock import patch

from src.utils.pylogger import RankedLogger


def _make_logger() -> RankedLogger:
    """Create a RankedLogger with rank set to 0."""
    logger = RankedLogger(name="test_logger")
    return logger


def test_rank_is_keyword_only():
    """log() must accept %-style format args positionally without rank
    consuming them (bug H17 fix).

    Before the fix, `rank` was a positional parameter and calling
    ``log(INFO, "value is %d", 42)`` would assign 42 to rank instead
    of treating it as a format argument.
    """
    logger = _make_logger()
    with patch(
        "src.utils.pylogger.rank_zero_only"
    ) as mock_rzo:
        mock_rzo.rank = 0
        # This must not raise TypeError or misinterpret 42 as rank.
        logger.log(logging.INFO, "value is %d", 42)


def test_rank_keyword_arg():
    """Passing rank as a keyword argument should still work."""
    logger = _make_logger()
    with patch(
        "src.utils.pylogger.rank_zero_only"
    ) as mock_rzo:
        mock_rzo.rank = 0
        logger.log(logging.INFO, "hello from rank 0", rank=0)
