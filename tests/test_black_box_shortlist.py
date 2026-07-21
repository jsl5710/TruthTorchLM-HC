"""Apply the §1 filter to the *real* TruthTorchLM methods.

Skipped when the full ML stack is unavailable, since it must import the actual classes.
Its job is to catch the single most consequential surprise in the shortlist: upstream
``SemanticEntropy`` requires sampled log-probabilities and is therefore **not** admissible
under the benchmark's text-only constraint. If a future upstream merge changes that flag,
this test tells us -- rather than a benchmark table quietly including a grey-box method.
"""

import pytest

pytest.importorskip(
    "torch", reason="requires the full TruthTorchLM dependency stack (torch/transformers/litellm)"
)

from TruthTorchLM.truth_methods import (  # noqa: E402
    NumSemanticSetUncertainty,
    SemanticEntropy,
    VerbalizedConfidence,
)
from TruthTorchLM.utils.access_level import AccessLevel, access_level  # noqa: E402


def test_semantic_entropy_is_grey_box_not_black_box():
    """The finding that reshapes the shortlist -- see protocol §1."""
    assert SemanticEntropy.REQUIRES_SAMPLED_LOGPROBS is True
    assert access_level(SemanticEntropy) is AccessLevel.GREY_BOX


def test_verbalized_confidence_is_black_box():
    assert access_level(VerbalizedConfidence) is AccessLevel.BLACK_BOX


def test_num_semantic_set_is_black_box():
    """The cluster-count consistency method needs sampled text only."""
    assert access_level(NumSemanticSetUncertainty) is AccessLevel.BLACK_BOX
