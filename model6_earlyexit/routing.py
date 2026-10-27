"""
routing.py
----------
Pure routing logic for Model 6.  No torch.nn, no file I/O.
Applied ONLY at prediction time; never called during training.

Usage
-----
    from routing import route_batch, route_single

    # outputs is the dict returned by EarlyExitModel.forward()
    decisions = route_batch(outputs, cfg["routing"])
    # decisions[i] = {"pred_class": str, "confidence": float,
    #                 "decided_by": str, "p_*": float, "c_*": float}
"""

import torch
from typing import Dict, List, Any


HEAD_ORDER_DEFAULT = ["no_feature", "dispersive", "cohesive", "noncohesive"]
CLASS_IDX = {"cohesive": 0, "dispersive": 1, "no_feature": 2, "noncohesive": 3}
IDX_CLASS = {v: k for k, v in CLASS_IDX.items()}


def route_batch(outputs: Dict[str, torch.Tensor], routing_cfg: dict) -> List[dict]:
    """
    Apply early-exit routing to a whole batch.

    Parameters
    ----------
    outputs      : dict from EarlyExitModel.forward()  (raw logits, on any device)
    routing_cfg  : the 'routing' section of config.yaml

    Returns
    -------
    list of per-tile decision dicts (length = batch size B)
    """
    priority = routing_cfg.get("priority_order", HEAD_ORDER_DEFAULT)
    resolution = routing_cfg.get("resolution", "first")   # "first" | "max"
    thresholds = routing_cfg["thresholds"]                 # {class_name: float}

    # Convert logits -> probabilities (all on CPU as numpy for readability)
    with torch.no_grad():
        p = {
            "no_feature" : torch.sigmoid(outputs["head_no_feature"]).cpu(),
            "dispersive" : torch.sigmoid(outputs["head_dispersive"]).cpu(),
            "cohesive"   : torch.sigmoid(outputs["head_cohesive"]).cpu(),
            "noncohesive": torch.sigmoid(outputs["head_noncohesive"]).cpu(),
        }
        concluder_probs = torch.softmax(outputs["concluder"], dim=1).cpu()  # [B, 4]

    B = concluder_probs.shape[0]
    decisions = []

    for i in range(B):
        tile_p = {k: float(v[i]) for k, v in p.items()}
        tile_c = {IDX_CLASS[j]: float(concluder_probs[i, j]) for j in range(4)}

        # Find all heads that cleared their threshold
        cleared = [cls for cls in priority if tile_p[cls] >= thresholds.get(cls, 0.7)]

        if cleared:
            if resolution == "max":
                winner = max(cleared, key=lambda c: tile_p[c])
            else:   # "first"
                winner = cleared[0]
            dec = {
                "pred_class" : winner,
                "confidence" : tile_p[winner],
                "decided_by" : f"head_{winner}",
            }
        else:
            # No head cleared -> concluder decides
            c_idx   = int(torch.argmax(concluder_probs[i]).item())
            c_class = IDX_CLASS[c_idx]
            dec = {
                "pred_class" : c_class,
                "confidence" : tile_c[c_class],
                "decided_by" : "concluder",
            }

        # Attach all raw probabilities for tile_scores.csv
        dec.update({f"p_{k}": tile_p[k] for k in tile_p})
        dec.update({f"c_{k}": tile_c[k] for k in tile_c})
        decisions.append(dec)

    return decisions


def route_single(outputs_single: Dict[str, Any], routing_cfg: dict) -> dict:
    """Convenience wrapper for a batch of size 1."""
    return route_batch(outputs_single, routing_cfg)[0]
