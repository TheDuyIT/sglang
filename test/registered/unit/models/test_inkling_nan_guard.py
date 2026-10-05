"""CPU unit tests for the NaN guard fix in InklingBatchDenseMLP._shared_scales.

Two tests:

1. TestNanGuardLogic — pure IEEE 754 logic; no sglang imports.  Confirms the
   original guard (``<= 0.0``) is a no-op for NaN and that the replacement
   (``not (x > 0.0)``) correctly catches NaN, negative, and zero.

2. TestSharedScalesNanGuard — calls ``_shared_scales`` directly with
   controlled inputs via a mock context, confirming the patched code returns
   ``input_scale = 1.0`` for NaN / negative / zero ``input_amax`` and the
   correct derived scale for a valid amax.

Run: python3 test/registered/unit/models/test_inkling_nan_guard.py
"""

import math
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="weekly", runner_config="cpu")


class TestNanGuardLogic(unittest.TestCase):
    """IEEE 754 correctness of the old vs new guard expression."""

    def _old_guard(self, v: float) -> bool:
        """Original broken condition: fires for negative/zero but not NaN."""
        return v <= 0.0

    def _new_guard(self, v: float) -> bool:
        """Patched condition: fires for NaN, negative, zero, and -inf."""
        return not (v > 0.0)

    def test_old_guard_misses_nan(self):
        self.assertFalse(self._old_guard(float("nan")))

    def test_old_guard_misses_neg_inf(self):
        # -inf <= 0.0 is True, so the old guard does catch -inf — only NaN escapes.
        self.assertTrue(self._old_guard(float("-inf")))

    def test_new_guard_catches_nan(self):
        self.assertTrue(self._new_guard(float("nan")))

    def test_new_guard_catches_negative(self):
        self.assertTrue(self._new_guard(-1.0))

    def test_new_guard_catches_zero(self):
        self.assertTrue(self._new_guard(0.0))

    def test_new_guard_catches_neg_inf(self):
        self.assertTrue(self._new_guard(float("-inf")))

    def test_new_guard_passes_positive(self):
        self.assertFalse(self._new_guard(1.0))
        self.assertFalse(self._new_guard(1e-10))
        self.assertFalse(self._new_guard(float("inf")))


class TestSharedScalesNanGuard(unittest.TestCase):
    """Calls _shared_scales directly with a mocked runtime context.

    Verifies the patched guard substitutes a neutral input_scale=1.0 when
    input_amax is NaN, negative, or zero, and preserves the correct derived
    value for a valid amax.
    """

    def setUp(self):
        # _shared_scales calls get_model() only to check load_format == "dummy".
        # Patch it at the module level so no real runtime context is needed.
        mock_model = MagicMock()
        mock_model.load_format = "auto"
        self._patcher = patch(
            "sglang.srt.models.inkling_common.dense_mlp.get_model",
            return_value=mock_model,
        )
        self._patcher.start()

        from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP

        # _shared_scales doesn't use self beyond the get_model() call above.
        self._shared_scales = InklingBatchDenseMLP._shared_scales
        self._fake_self = types.SimpleNamespace()

    def tearDown(self):
        self._patcher.stop()

    def _call(self, amax_val: float) -> torch.Tensor:
        """Call _shared_scales with a constant scale2 and the given amax."""
        scale2 = torch.ones(1)
        input_amax = torch.tensor([amax_val], dtype=torch.float32)
        _, input_scale = self._shared_scales(self._fake_self, scale2, input_amax)
        return input_scale

    def test_nan_amax_returns_neutral_scale(self):
        result = self._call(float("nan"))
        self.assertTrue(torch.isfinite(result), f"expected finite, got {result}")
        self.assertAlmostEqual(result.item(), 1.0)

    def test_negative_amax_returns_neutral_scale(self):
        result = self._call(-1.0)
        self.assertAlmostEqual(result.item(), 1.0)

    def test_zero_amax_returns_neutral_scale(self):
        result = self._call(0.0)
        self.assertAlmostEqual(result.item(), 1.0)

    def test_valid_amax_returns_derived_scale(self):
        amax = 6.0 * 448.0  # exactly 1.0 after division
        result = self._call(amax)
        self.assertAlmostEqual(result.item(), 1.0)

    def test_valid_amax_small_returns_correct_scale(self):
        amax = 448.0
        expected = 448.0 / (6.0 * 448.0)
        result = self._call(amax)
        self.assertAlmostEqual(result.item(), expected, places=6)

    def test_nan_amax_result_finite(self):
        # Belt-and-suspenders: the returned input_scale must be torch.isfinite.
        result = self._call(float("nan"))
        self.assertFalse(torch.isnan(result), "input_scale must not be NaN")
        self.assertFalse(torch.isinf(result), "input_scale must not be inf")


if __name__ == "__main__":
    unittest.main()
