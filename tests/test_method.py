from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from facetprint.construction import extract_words, recombine_word_positions
from facetprint.quant_lora import FacetPrintLinear, fake_quant_symmetric
from facetprint.verification import accept, exact_match


class ConstructionTests(unittest.TestCase):
    def test_recombination_is_seeded_and_position_preserving(self):
        candidates = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        target_a, trace_a = recombine_word_positions(candidates, 3, 43)
        target_b, trace_b = recombine_word_positions(candidates, 3, 43)
        self.assertEqual(target_a, target_b)
        self.assertEqual(trace_a, trace_b)
        for position, word in enumerate(target_a):
            self.assertIn(word, {candidate[position] for candidate in candidates})

    def test_recombination_rejects_short_candidate(self):
        with self.assertRaises(ValueError):
            recombine_word_positions([["a", "b"], ["c"]], 2, 43)

    def test_word_extraction_matches_formal_implementation(self):
        self.assertEqual(extract_words("alpha beta-ray's 42"), ["alpha", "beta-ray's"])


class QuantizationTests(unittest.TestCase):
    def test_fake_quant_has_straight_through_gradient(self):
        tensor = torch.tensor([[-1.2, -0.1, 0.8, 1.4]], requires_grad=True)
        output = fake_quant_symmetric(tensor, bits=4, group_size=None)
        output.sum().backward()
        self.assertTrue(torch.equal(tensor.grad, torch.ones_like(tensor)))
        self.assertEqual(output.shape, tensor.shape)

    def test_bf16_view_matches_merged_layer(self):
        torch.manual_seed(7)
        base = nn.Linear(4, 3, bias=True)
        layer = FacetPrintLinear(base, rank=2, alpha=4, group_size=2)
        with torch.no_grad():
            layer.lora_b.normal_()
        inputs = torch.randn(2, 4)
        before = layer(inputs)
        after = layer.merged_linear()(inputs)
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-5))


class VerificationTests(unittest.TestCase):
    def test_exact_match_only_strips_outer_whitespace(self):
        self.assertTrue(exact_match("  Target\n", "Target"))
        self.assertFalse(exact_match("target", "Target"))
        self.assertFalse(exact_match("Tar  get", "Tar get"))

    def test_q3_p1_rule(self):
        self.assertTrue(accept([False, True], required_matches=1))
        self.assertFalse(accept([False, False, False], required_matches=1))


if __name__ == "__main__":
    unittest.main()
