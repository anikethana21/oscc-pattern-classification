import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import unittest
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, "D:/Project/model4_cascade")
sys.path.insert(0, "D:/Project/model6_earlyexit")

from routing import route_batch
from sort_tiles import sort_tiles


class TestModel6RoutingAndOutputs(unittest.TestCase):

    def test_routing_first_wins(self):
        """When multiple heads clear, resolution='first' picks the earliest in priority order."""
        outputs = {
            "head_no_feature": torch.tensor([2.0]),   # sigmoid(2.0) ~ 0.88
            "head_dispersive": torch.tensor([3.0]),   # sigmoid(3.0) ~ 0.95
            "head_cohesive": torch.tensor([-2.0]),
            "head_noncohesive": torch.tensor([-2.0]),
            "concluder": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        }
        cfg = {
            "priority_order": ["no_feature", "dispersive", "cohesive", "noncohesive"],
            "resolution": "first",
            "thresholds": {"no_feature": 0.70, "dispersive": 0.70, "cohesive": 0.70, "noncohesive": 0.70}
        }
        dec = route_batch(outputs, cfg)[0]
        self.assertEqual(dec["pred_class"], "no_feature")
        self.assertEqual(dec["decided_by"], "head_no_feature")
        self.assertGreaterEqual(dec["confidence"], 0.70)

    def test_routing_max_wins(self):
        """When multiple heads clear, resolution='max' picks the one with highest probability."""
        outputs = {
            "head_no_feature": torch.tensor([1.0]),   # sigmoid(1.0) ~ 0.731
            "head_dispersive": torch.tensor([2.5]),   # sigmoid(2.5) ~ 0.924 (highest)
            "head_cohesive": torch.tensor([-1.0]),
            "head_noncohesive": torch.tensor([-1.0]),
            "concluder": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        }
        cfg = {
            "priority_order": ["no_feature", "dispersive", "cohesive", "noncohesive"],
            "resolution": "max",
            "thresholds": {"no_feature": 0.70, "dispersive": 0.70, "cohesive": 0.70, "noncohesive": 0.70}
        }
        dec = route_batch(outputs, cfg)[0]
        self.assertEqual(dec["pred_class"], "dispersive")
        self.assertEqual(dec["decided_by"], "head_dispersive")
        self.assertAlmostEqual(dec["confidence"], torch.sigmoid(torch.tensor(2.5)).item(), places=4)

    def test_routing_concluder_fallback(self):
        """When no head clears its threshold, the 4-class concluder decides."""
        outputs = {
            "head_no_feature": torch.tensor([-1.0]),   # ~0.26
            "head_dispersive": torch.tensor([0.0]),    # 0.50
            "head_cohesive": torch.tensor([0.2]),      # ~0.55 < 0.70
            "head_noncohesive": torch.tensor([-2.0]),  # ~0.11
            "concluder": torch.tensor([[0.5, 0.2, 0.1, 2.5]]),  # noncohesive has highest logit
        }
        cfg = {
            "priority_order": ["no_feature", "dispersive", "cohesive", "noncohesive"],
            "resolution": "first",
            "thresholds": {"no_feature": 0.70, "dispersive": 0.70, "cohesive": 0.70, "noncohesive": 0.70}
        }
        dec = route_batch(outputs, cfg)[0]
        self.assertEqual(dec["pred_class"], "noncohesive")
        self.assertEqual(dec["decided_by"], "concluder")
        self.assertGreater(dec["confidence"], 0.0)

    def test_routing_tie_case(self):
        """In exact probability tie between cleared heads, first in priority wins."""
        outputs = {
            "head_no_feature": torch.tensor([2.0]),
            "head_dispersive": torch.tensor([2.0]),
            "head_cohesive": torch.tensor([-1.0]),
            "head_noncohesive": torch.tensor([-1.0]),
            "concluder": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        }
        cfg = {
            "priority_order": ["dispersive", "no_feature", "cohesive", "noncohesive"],
            "resolution": "max",
            "thresholds": {"no_feature": 0.70, "dispersive": 0.70, "cohesive": 0.70, "noncohesive": 0.70}
        }
        dec = route_batch(outputs, cfg)[0]
        self.assertEqual(dec["pred_class"], "dispersive")

    def test_sorting_order(self):
        """Verify sort_tiles orders by class and confidence correctly."""
        data = {
            "tile_id": ["t1", "t2", "t3", "t4"],
            "pred_class": ["cohesive", "cohesive", "dispersive", "dispersive"],
            "confidence": [0.65, 0.95, 0.40, 0.88],
        }
        df = pd.DataFrame(data)
        tmp_in = "D:/Project/model6_earlyexit/tests/temp_test_scores.csv"
        tmp_out = "D:/Project/model6_earlyexit/tests/temp_test_sorted.csv"
        df.to_csv(tmp_in, index=False)

        sorted_df = sort_tiles(tmp_in, tmp_out, ascending=False)
        coh_confs = sorted_df[sorted_df["pred_class"] == "cohesive"]["confidence"].tolist()
        self.assertEqual(coh_confs, [0.95, 0.65])
        disp_confs = sorted_df[sorted_df["pred_class"] == "dispersive"]["confidence"].tolist()
        self.assertEqual(disp_confs, [0.88, 0.40])

        sorted_asc = sort_tiles(tmp_in, tmp_out, ascending=True)
        coh_asc = sorted_asc[sorted_asc["pred_class"] == "cohesive"]["confidence"].tolist()
        self.assertEqual(coh_asc, [0.65, 0.95])

        if os.path.exists(tmp_in): os.remove(tmp_in)
        if os.path.exists(tmp_out): os.remove(tmp_out)


if __name__ == "__main__":
    unittest.main()
