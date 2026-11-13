"""UNDEPLOYED proposal: conservative v7 observation subclaim coalescing.

Never use this on proposal or state-blind extraction responses, and never mutate
frozen v7. It only combines separately validated assessments with the SAME exact
caller-owned unit ID, retaining the least factual support. It cannot repair
missing IDs, uncertain support, invented text, schema errors, or bad semantics.
"""
from copy import deepcopy

from .judge_v2 import _claims, _common, _exact, resolve_claim_ids, validate_observations


def coalesce_duplicate_observation_units(value, units):
    _exact(value, ("evidence", "claims", "assessable", "issues"))
    _common(value)
    if not isinstance(value["claims"], list):
        raise ValueError("Claims must be a list")
    groups = {}
    for claim in value["claims"]:
        _exact(claim, ("claim_id", "evidence", "kind", "support"))
        index = claim["claim_id"]
        if type(index) is not int or not 0 <= index < len(units):
            raise ValueError("Invalid source unit ID")
        # Validate every subclaim BEFORE minimum aggregation. In particular,
        # uncertainty cannot be hidden by some other unsupported subclaim.
        _claims([{key: item for key, item in claim.items() if key != "claim_id"}
                 | {"quote": units[index]["text"]}])
        groups.setdefault(index, []).append(claim)
    if sorted(groups) != list(range(len(units))):
        raise ValueError("Missing source units cannot be repaired by duplicate claims")
    result = deepcopy(value)
    result["claims"] = []
    for index, subclaims in sorted(groups.items()):
        if len(subclaims) == 1:
            result["claims"].append(deepcopy(subclaims[0]))
            continue
        factual = [claim for claim in subclaims if claim["kind"] != "intention"]
        if factual:
            support = min(factual, key=lambda claim: {"unsupported": 0, "partial": 2, "supported": 4}[claim["support"]])["support"]
            kind = "other"  # V7's independent semantics governs factual type.
        else:
            support, kind = "not_applicable", "intention"
        result["claims"].append(dict(claim_id=index, kind=kind, support=support,
            evidence=" | ".join(claim["evidence"] for claim in subclaims)))
        result["issues"].append(f"Research v8 normalization: combined {len(subclaims)} subclaims for exact unit {index}; least factual support={support}.")
    validate_observations(resolve_claim_ids(result, units))
    return result
