"""
graph_metrics.py — compute ELsEA-style edge weights for DistEA.

Metrics:
  r2f  (relation-to-functionality):  P(unique tail | relation)
  r2if (inverse functionality):       P(unique head | relation, reversed)

Result cached to {output_dir}/KGs_edge_info.json and KGs_node_info.json.
"""

import json
import logging
import os
from collections import defaultdict


def _compute_r2f_r2if(triples):
    """
    Compute r2f and r2if for a KG's triples.

    r2f[r]  = |{t : (h, r, t) in G}| / |{(h, r, t) in G}|
    r2if[r] = |{h : (h, r, t) in G}| / |{(h, r, t) in G}|

    Returns two dicts: r2f, r2if  (relation_id → float in [0,1])
    """
    rel_heads = defaultdict(set)
    rel_tails = defaultdict(set)
    rel_count = defaultdict(int)

    for h, r, t in triples:
        rel_heads[r].add(h)
        rel_tails[r].add(t)
        rel_count[r] += 1

    r2f = {}
    r2if = {}
    for r, cnt in rel_count.items():
        r2f[r] = len(rel_tails[r]) / cnt
        r2if[r] = len(rel_heads[r]) / cnt

    return r2f, r2if


def _compute_degree(triples, entity_set):
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    for h, _r, t in triples:
        out_deg[h] += 1
        in_deg[t] += 1
    return (
        {e: in_deg[e] for e in entity_set},
        {e: out_deg[e] for e in entity_set},
    )


def graph_metrics_main(data):
    """
    Compute and cache graph metrics to data.output_dir.

    Writes:
        {output_dir}/KGs_edge_info.json   — r2f / r2if per relation
        {output_dir}/KGs_node_info.json   — in/out degree per entity
    """
    os.makedirs(data.output_dir, exist_ok=True)

    kg1_triples = data.kg1_triples
    kg2_triples = data.kg2_triples

    r2f1, r2if1 = _compute_r2f_r2if(kg1_triples)
    r2f2, r2if2 = _compute_r2f_r2if(kg2_triples)

    edge_info = {
        "KG1": {
            "r2f": {str(k): v for k, v in r2f1.items()},
            "r2if": {str(k): v for k, v in r2if1.items()},
        },
        "KG2": {
            "r2f": {str(k): v for k, v in r2f2.items()},
            "r2if": {str(k): v for k, v in r2if2.items()},
        },
    }
    edge_path = os.path.join(data.output_dir, "KGs_edge_info.json")
    with open(edge_path, "w") as f:
        json.dump(edge_info, f)
    logging.info(f"[graph_metrics] Edge info written to {edge_path}")

    # Node-level degree info
    kg1_ents = set(data.kg1_entities)
    kg2_ents = set(data.kg2_entities)
    in1, out1 = _compute_degree(kg1_triples, kg1_ents)
    in2, out2 = _compute_degree(kg2_triples, kg2_ents)

    node_info = {
        "KG1": {
            "in_degree": {str(k): v for k, v in in1.items()},
            "out_degree": {str(k): v for k, v in out1.items()},
        },
        "KG2": {
            "in_degree": {str(k): v for k, v in in2.items()},
            "out_degree": {str(k): v for k, v in out2.items()},
        },
    }
    node_path = os.path.join(data.output_dir, "KGs_node_info.json")
    with open(node_path, "w") as f:
        json.dump(node_info, f)
    logging.info(f"[graph_metrics] Node info written to {node_path}")
