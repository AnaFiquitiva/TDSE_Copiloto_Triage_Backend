import unittest

from src import metrics


class TestSubTriageRate(unittest.TestCase):
    def test_no_sub_triage(self):
        pairs = [(1, 1), (2, 2), (3, 2)]  # el ultimo es sobre-triage, no sub-triage
        self.assertEqual(metrics.sub_triage_rate(pairs), 0.0)

    def test_pure_sub_triage(self):
        pairs = [(1, 2), (1, 3)]  # subestima en 1 y en 2 niveles
        self.assertAlmostEqual(metrics.sub_triage_rate(pairs), 1.5)

    def test_empty(self):
        self.assertEqual(metrics.sub_triage_rate([]), 0.0)


class TestWeightedKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        pairs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        self.assertAlmostEqual(metrics.weighted_kappa(pairs), 1.0)

    def test_some_disagreement_is_less_than_one(self):
        pairs = [(1, 1), (2, 2), (3, 4), (4, 4), (5, 5)]
        kappa = metrics.weighted_kappa(pairs)
        self.assertLess(kappa, 1.0)
        self.assertGreater(kappa, 0.0)


class TestSensitivity(unittest.TestCase):
    def test_all_high_risk_detected(self):
        pairs = [(1, 1), (2, 2), (3, 3)]
        self.assertEqual(metrics.sensitivity_high_risk(pairs), 1.0)

    def test_missed_high_risk_case(self):
        pairs = [(1, 3), (2, 2)]  # el primero subestima I hacia III
        self.assertEqual(metrics.sensitivity_high_risk(pairs), 0.5)

    def test_no_high_risk_cases_is_nan(self):
        pairs = [(3, 3), (4, 4)]
        self.assertNotEqual(metrics.sensitivity_high_risk(pairs), metrics.sensitivity_high_risk(pairs))


class TestAbstentionMetrics(unittest.TestCase):
    def test_correct_and_improper_abstention(self):
        should_abstain = [True, True, False, False]
        actually_abstained = [True, False, False, True]
        result = metrics.abstention_metrics(should_abstain, actually_abstained)
        self.assertEqual(result["tasa_abstencion_correcta"], 0.5)
        self.assertEqual(result["tasa_abstencion_indebida"], 0.5)


class TestCoverageAndPairedS(unittest.TestCase):
    def test_coverage_counts_non_abstained(self):
        self.assertEqual(metrics.coverage([1, None, 3, None]), 0.5)

    def test_paired_sub_triage_uses_only_cases_both_answer(self):
        ref = [1, 1, 2, 3]
        a = [2, None, 2, 3]  # A se abstiene en el segundo caso
        b = [1, 3, None, 4]  # B se abstiene en el tercero
        result = metrics.paired_sub_triage(ref, a, b)
        self.assertEqual(result["n"], 2)  # casos 1 y 4
        self.assertAlmostEqual(result["S_a"], 0.5)  # (1 + 0) / 2
        self.assertAlmostEqual(result["S_b"], 0.5)  # (0 + 1) / 2


class TestOverTriageAndAgreement(unittest.TestCase):
    def test_over_triage_is_symmetric_to_sub_triage(self):
        pairs = [(3, 1), (2, 2), (1, 2)]
        self.assertAlmostEqual(metrics.over_triage_rate(pairs), 2 / 3)
        self.assertAlmostEqual(metrics.sub_triage_rate(pairs), 1 / 3)

    def test_exact_agreement(self):
        self.assertEqual(metrics.exact_agreement([(1, 1), (2, 3)]), 0.5)


class TestRankingMetrics(unittest.TestCase):
    def test_reciprocal_rank(self):
        self.assertEqual(metrics.reciprocal_rank(["A", "B", "C"], "B"), 0.5)
        self.assertEqual(metrics.reciprocal_rank(["A", "B"], "Z"), 0.0)

    def test_mrr_and_recall_at_1(self):
        ranked = [["A", "B"], ["B", "A"], ["C", "D"]]
        relevant = ["A", "A", "A"]
        rr = [metrics.reciprocal_rank(r, rel) for r, rel in zip(ranked, relevant)]
        self.assertAlmostEqual(metrics.mean_reciprocal_rank(rr), (1 + 0.5 + 0) / 3)
        self.assertAlmostEqual(metrics.recall_at_1(ranked, relevant), 1 / 3)


class TestBootstrapCI(unittest.TestCase):
    @staticmethod
    def _mean(values):
        return sum(values) / len(values)

    def test_same_seed_is_reproducible(self):
        values = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        first = metrics.bootstrap_ci(values, self._mean, n=500, seed=7)
        second = metrics.bootstrap_ci(values, self._mean, n=500, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first["ci_low"], first["estimate"])
        self.assertGreaterEqual(first["ci_high"], first["estimate"])

    def test_works_with_pairs(self):
        pairs = [(1, 2), (2, 2), (1, 1), (3, 5)]
        result = metrics.bootstrap_ci(pairs, metrics.sub_triage_rate, n=200)
        self.assertAlmostEqual(result["estimate"], 0.75)

    def test_constant_sample_has_degenerate_interval(self):
        result = metrics.bootstrap_ci([1, 1, 1], self._mean, n=100)
        self.assertEqual((result["ci_low"], result["ci_high"]), (1.0, 1.0))


class TestMcNemarExact(unittest.TestCase):
    def test_known_value_zero_and_five(self):
        self.assertAlmostEqual(metrics.mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(metrics.mcnemar_exact(5, 0), 0.0625)

    def test_no_discordant_pairs_gives_one(self):
        self.assertEqual(metrics.mcnemar_exact(0, 0), 1.0)

    def test_balanced_discordance_is_not_significant(self):
        self.assertEqual(metrics.mcnemar_exact(3, 3), 1.0)

    def test_known_value_one_and_nine(self):
        # 2 * (C(10,0) + C(10,1)) / 2**10 = 22 / 1024
        self.assertAlmostEqual(metrics.mcnemar_exact(1, 9), 22 / 1024)


if __name__ == "__main__":
    unittest.main()
