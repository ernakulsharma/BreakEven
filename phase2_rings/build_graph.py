"""Build the transaction/entity graph from IEEE-CIS's own linking columns.

Nodes are transactions and entities (card fingerprint, address, device, email
domain, uid). Edges connect a transaction to each entity it touches. Two
transactions are in the same component when they share any entity, which is
exactly the definition of an abuse ring: one operator reusing cards, devices,
drop addresses or mailboxes across many orders.

The single thing that makes or breaks this graph is HUB PRUNING.
`P_emaildomain == "gmail.com"` appears on ~230k of the 590k IEEE-CIS rows. If
you keep it as a linking edge, every gmail user on the internet joins one
component and the graph collapses into a single blob with ~90% of transactions
in it. Entities above a degree threshold carry no identifying information and
are dropped before components are computed. This is why a naive
`nx.connected_components` on this data returns a useless answer.

We use scipy sparse + csgraph rather than networkx: 590k transactions and
~1M entity nodes is well past the point where a Python-object graph fits in
3.9GB, let alone runs fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common.features import add_entity_key_columns

# Hub limits must scale with the dataset. A fixed limit of 500 is strict on a
# 30k-row sample and useless on 590k rows, where the same field has 20x the
# degree. Effective limit = max(floor, rate * n_rows).
DEFAULT_HUB_RATE = {"card": 0.001, "uid": 0.001, "addr": 0.001,
                    "device": 0.001, "email": 0.0005}
DEFAULT_HUB_FLOOR = {"card": 50, "uid": 50, "addr": 50, "device": 50, "email": 20}

# A field is only usable for linking if it is IDENTIFYING. If its average
# degree exceeds this, it is a category label rather than an entity, and
# joining on it fuses unrelated transactions into one blob.
#
# This is not hypothetical: IEEE-CIS `addr1` has ~332 distinct values across
# 590,540 rows (average degree ~1,780). It is a coarse region/billing-zone
# code, NOT a street address, so it cannot identify a drop address no matter
# how the graph is built. `P_emaildomain` is likewise only the domain, so
# "gmail.com" links 40% of the dataset. Both are dropped automatically here,
# and both were listed as ring-linking fields in the original plan — the data
# says otherwise.
MAX_AVG_DEGREE_TO_BE_IDENTIFYING = 50

# Two transactions are only linked through a shared entity if they fall within
# this many seconds of each other. 7 days: long enough to span a reshipping
# operation or a card-testing campaign, short enough that a region code reused
# across a year does not fuse unrelated traffic into one component.
DEFAULT_LINK_WINDOW_SECONDS = 7 * 24 * 3600

# Retained for callers that want to pin explicit absolute limits.
DEFAULT_HUB_LIMITS: dict[str, int] = {}

# Placeholder values that mean "unknown", not "the same entity".
NULL_TOKENS = {"na", "", "nan", "none", "na|na", "na|na|na", "na|na|na|na"}


@dataclass
class EntityGraph:
    tx_ids: np.ndarray                  # transaction id per transaction node
    labels: np.ndarray                  # component id per transaction node
    n_components: int
    edges: list[tuple[int, int, str, str]] = field(default_factory=list)
    entity_degree: dict[str, dict[str, int]] = field(default_factory=dict)
    pruned_hubs: dict[str, int] = field(default_factory=dict)
    dropped_namespaces: dict[str, dict] = field(default_factory=dict)
    hub_limits: dict[str, int] = field(default_factory=dict)
    link_window_seconds: float | None = None
    frame: pd.DataFrame | None = None


def _is_null_key(v: str) -> bool:
    s = str(v).strip().lower()
    return s in NULL_TOKENS or set(s.split("|")) <= {"na", "nan", ""}


def build_graph(
    df: pd.DataFrame,
    namespaces=("card", "addr", "device", "email", "uid"),
    hub_limits: dict[str, int] | None = None,
    keep_edges: bool = True,
    link_window_seconds: float | None = DEFAULT_LINK_WINDOW_SECONDS,
) -> EntityGraph:
    """Return connected components over the transaction/entity bipartite graph."""
    df = df.reset_index(drop=True)
    if "card_key" not in df.columns:
        df = add_entity_key_columns(df)

    n_tx = len(df)
    # Size-adaptive hub limits, with explicit caller overrides applied last.
    effective_limits = {
        ns: max(DEFAULT_HUB_FLOOR.get(ns, 50), int(DEFAULT_HUB_RATE.get(ns, 0.001) * n_tx))
        for ns in namespaces
    }
    effective_limits.update(DEFAULT_HUB_LIMITS)
    effective_limits.update(hub_limits or {})
    hub_limits = effective_limits
    tx_ids = (df["TransactionID"].to_numpy() if "TransactionID" in df.columns
              else np.arange(n_tx))

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    next_node = n_tx
    entity_degree: dict[str, dict[str, int]] = {}
    pruned: dict[str, int] = {}
    dropped_namespaces: dict[str, dict] = {}
    edges: list[tuple[int, int, str, str]] = []

    for ns in namespaces:
        keycol = f"{ns}_key"
        if keycol not in df.columns:
            continue
        keys = df[keycol].astype(str).to_numpy()
        valid = np.array([not _is_null_key(k) for k in keys])

        codes, uniques = pd.factorize(pd.Series(np.where(valid, keys, None)))
        counts = np.bincount(codes[codes >= 0], minlength=len(uniques))

        # Identifiability guard: reject the whole field if it behaves like a
        # category rather than an entity. Pruning individual hubs cannot save
        # a field where *every* value is a hub.
        n_valid, n_distinct = int(valid.sum()), len(uniques)
        avg_degree = n_valid / max(n_distinct, 1)
        if n_distinct and avg_degree > MAX_AVG_DEGREE_TO_BE_IDENTIFYING:
            dropped_namespaces[ns] = {
                "distinct_values": n_distinct,
                "avg_degree": round(float(avg_degree), 1),
                "reason": "field is categorical, not identifying — linking on it "
                          "would fuse unrelated transactions",
            }
            entity_degree[ns] = {}
            continue

        limit = hub_limits.get(ns, 10**9)
        is_hub = counts > limit
        pruned[ns] = int(is_hub.sum())

        keep = (codes >= 0) & ~is_hub[np.clip(codes, 0, None)]
        if not keep.any():
            entity_degree[ns] = {}
            continue

        kept_codes = codes[keep]
        kept_idx = np.nonzero(keep)[0]
        entity_degree[ns] = {str(uniques[c]): int(counts[c]) for c in np.unique(kept_codes)}

        if link_window_seconds is None or "TransactionDT" not in df.columns:
            # Star topology: every transaction on an entity joins one entity node.
            uniq_kept, inverse = np.unique(kept_codes, return_inverse=True)
            rows.append(kept_idx)
            cols.append(next_node + inverse)
            next_node += len(uniq_kept)
        else:
            # Time-windowed chaining. An abuse ring is a BURST: the same drop
            # address, device or card is hit repeatedly over hours or days.
            # A shared value reused months apart is co-incidence, not collusion.
            #
            # So instead of joining every transaction on an entity into one
            # star (which chains a year of unrelated activity into a single
            # giant component), we sort each entity's transactions by time and
            # link only CONSECUTIVE pairs falling inside the window. Long gaps
            # break the chain naturally, with no bucket-boundary artifacts.
            dt = df["TransactionDT"].to_numpy()[kept_idx].astype("float64")
            order = np.lexsort((dt, kept_codes))
            sk, sd, si = kept_codes[order], dt[order], kept_idx[order]
            same_entity = sk[1:] == sk[:-1]
            within_window = (sd[1:] - sd[:-1]) <= link_window_seconds
            link = same_entity & within_window
            if link.any():
                rows.append(si[:-1][link])
                cols.append(si[1:][link])

        if keep_edges:
            # Transaction -> entity incidences, kept for explainability. Ring
            # members come from the components, so we never materialise the
            # O(k^2) clique a naive expansion would produce.
            for tx_idx, ent_code in zip(kept_idx, kept_codes):
                edges.append((int(tx_idx), int(ent_code), ns, str(uniques[ent_code])))

    if not rows:
        return EntityGraph(tx_ids=tx_ids, labels=np.arange(n_tx), n_components=n_tx,
                           entity_degree={}, pruned_hubs=pruned,
                           dropped_namespaces=dropped_namespaces,
                           hub_limits=hub_limits,
                           link_window_seconds=link_window_seconds, frame=df)

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    n_nodes = next_node
    adj = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n_nodes, n_nodes))
    adj = adj + adj.T

    n_comp, labels_all = connected_components(adj.tocsr(), directed=False)
    labels = labels_all[:n_tx]

    return EntityGraph(
        tx_ids=tx_ids, labels=labels, n_components=int(n_comp), edges=edges,
        entity_degree=entity_degree, pruned_hubs=pruned,
        dropped_namespaces=dropped_namespaces, hub_limits=hub_limits,
        link_window_seconds=link_window_seconds, frame=df,
    )


def component_sizes(graph: EntityGraph) -> pd.Series:
    return pd.Series(graph.labels).value_counts()


def graph_summary(graph: EntityGraph) -> dict:
    sizes = component_sizes(graph)
    multi = sizes[sizes > 1]
    return {
        "n_transactions": int(len(graph.labels)),
        "n_components": int(graph.n_components),
        "n_multi_transaction_components": int(len(multi)),
        "largest_component": int(sizes.max()) if len(sizes) else 0,
        "share_in_largest": float(sizes.max() / len(graph.labels)) if len(sizes) else 0.0,
        "median_multi_component_size": float(multi.median()) if len(multi) else 0.0,
        "link_window_seconds": graph.link_window_seconds,
        "pruned_hub_entities": graph.pruned_hubs,
        "hub_limits_applied": graph.hub_limits,
        "dropped_non_identifying_fields": graph.dropped_namespaces,
    }
