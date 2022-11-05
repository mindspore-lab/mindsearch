import unittest
from mindsearch.evaluation.metrics_eval import compute_metrics


class ComputeMetrics(unittest.TestCase):
    
    def test_compute_mrr(self):
        qp_ids_gold = {"q1": ["p1"], "q2": ["p2"]}
        qp_ids_candi = {"q1": ["p2", "p3", "p1"], "q2": ["p2", "p1", "p3"]}
        
        results = compute_metrics(qp_ids_gold, qp_ids_candi)
        self.assertEqual(results[0][1], 0.5)
        self.assertEqual(results[2][1], 2 / 3)
