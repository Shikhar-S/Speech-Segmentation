"""CMU L2Arctic Dataset Package.

This package contains data modules for CMU Arctic and L2-ARCTIC corpora.
"""

from .l1_classification import CmuL2ArcticL1Classification
from .ipa_data import CmuL2ArcticIPADataModule

__all__ = ["CmuL2ArcticL1Classification", "CmuL2ArcticIPADataModule"]

