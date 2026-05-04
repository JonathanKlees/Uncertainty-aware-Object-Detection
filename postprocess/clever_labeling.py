from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Optional, List, Tuple


Counts = Dict[str, int]
Dataset = Dict[str, Counts]


def _normalize_from_counts(counts: Mapping[str, int], classes: Sequence[str]) -> Dict[str, float]:
    n = sum(counts.get(c, 0) for c in classes)
    if n <= 0:
        # return uniform if empty
        K = len(classes)
        return {c: 1.0 / K for c in classes} if K else {}
    return {c: counts.get(c, 0) / n for c in classes}


def _round_probs_to_counts(probs: Mapping[str, float], total: int) -> Counts:
    """
    Largest-remainder rounding that preserves total exactly.
    
    Note: This is a thin wrapper for backward compatibility.
    Use smoothing.probs_to_counts for new code.
    """
    from postprocess.smoothing import probs_to_counts
    # Get support from input keys
    support = list(probs.keys())
    result = probs_to_counts(probs, total, support)
    return result


def _argmax_prob(probs: Mapping[str, float], classes: Sequence[str]) -> str:
    best_c = classes[0]
    best_v = -1.0
    for c in classes:
        v = float(probs.get(c, 0.0))
        if v > best_v:
            best_v = v
            best_c = c
    return best_c


def cleverlabel(
    biased_input: Counts | Dict[str, float],
    *,
    proposal_class: str,
    classes: Optional[Sequence[str]] = None,
    # Bias Correction (BC) parameters
    delta: float = 0.1,
    one_star: float = 0.99,
    avoid_overcorrection_threshold: float = 0,
    # Class Blending (CB) parameters
    mu: float = 0.75,
    transition_c: Optional[Mapping[str, Sequence[float]]] = None,
    apply_bc: bool = True,
    apply_cb: bool = True,
    return_counts: bool = False,
    input_probs: bool = False,
    pseudo_count_scale: int = 1000,

) -> Counts | Dict[str, float]:
    """
    Counts-in / prob or counts out CleverLabel for ONE entry.

    Inputs:
      biased_input: dict[class -> count] or dict[class -> prob] depending on input_probs.
      proposal_class: rho_x (the proposed class for this entry).
      classes: fixed class order; if None, inferred from biased_input keys and proposal_class.
      transition_c: class->row of length K encoding c(hat,k) for Class Blending.
                    If None, defaults to identity (no blending effect).
      input_probs: If True, biased_input is treated as probabilities (not counts).
      pseudo_count_scale: When input_probs=True and return_counts=True, use this as total N.

    Outputs:
      corrected_counts or corrected_probs depending on return_counts.
    """
    if classes is None:
        cls = set(biased_input.keys())
        cls.add(proposal_class)
        classes = sorted(cls)
    else:
        classes = list(classes)

    K = len(classes)
    if K == 0:
        return {}

    if input_probs:
        # Input is already probabilities
        p_b = {c: float(biased_input.get(c, 0.0)) for c in classes}
        # Normalize just in case
        s = sum(p_b.values())
        if s > 0:
            p_b = {c: v / s for c, v in p_b.items()}
        else:
            p_b = {c: 1.0 / K for c in classes}
        N = pseudo_count_scale  # Use scale for return_counts conversion
        # For BC, estimate acceptance rate from probs directly
        n_prop_ratio = p_b.get(proposal_class, 0.0)
    else:
        # Input is counts
        biased_counts = biased_input
        N = sum(biased_counts.get(c, 0) for c in classes)
        if N <= 0:
            return {}
        # Start from biased probabilities (MLE from counts)
        p_b = _normalize_from_counts(biased_counts, classes)
        n_prop_ratio = biased_counts.get(proposal_class, 0) / N

    # -----------------------
    # Bias Correction (BC)
    # -----------------------
    # Model assumption from paper: annotated class == proposal iff proposal was accepted.
    # So acceptance rate A is estimated as count(proposal)/N (or directly from probs).
    if apply_bc:
        A_hat = n_prop_ratio  # estimated acceptance probability A

        if A_hat < avoid_overcorrection_threshold:
            # data already shows ambiguity do not remove too much
            p_curr = p_b
        else:
            # Estimate B = P(L^x = rho_x) ≈ (A - δ)/(1* - δ), clamp to [0,1].
            denom = (one_star - delta)
            if denom <= 1e-12:
                raise ValueError("Invalid parameters: need one_star > delta")
            B = (A_hat - delta) / denom
            B = max(0.0, min(1.0, B))

            # Estimate reject-case conditional distribution from non-proposal labels:
            # p_rej(k) ≈ p_b(k) / (1 - p_b(proposal)), k != proposal
            rej_mass = 1.0 - p_b.get(proposal_class, 0.0)
            if rej_mass <= 1e-12:
                # Everyone accepted proposal -> all mass goes to proposal after correction
                p_u = {c: 0.0 for c in classes}
                p_u[proposal_class] = 1.0
            else:
                p_rej = {c: 0.0 for c in classes}
                for c in classes:
                    if c == proposal_class:
                        continue
                    p_rej[c] = p_b.get(c, 0.0) / rej_mass

                # Reconstruct unbiased distribution:
                # P(L^x=rho)=B, and for k!=rho: P(L^x=k)=(1-B)*P(L_b^x=k | L^x != rho).
                p_u = {c: (1.0 - B) * p_rej.get(c, 0.0) for c in classes}
                p_u[proposal_class] = B

            # replace current distribution with bias-corrected one
            p_curr = p_u
    else:
        p_curr = p_b

    # -----------------------
    # Class Blending (CB)
    # -----------------------
    if apply_cb:
        # choose hat_k as argmax over current probs (paper uses the most likely class). 
        hat_k = _argmax_prob(p_curr, classes)

        # build c(hat_k, ·)
        if transition_c is None:
            # identity row (no blending unless mu < 1 and identity is used)
            c_row = [1.0 if classes[i] == hat_k else 0.0 for i in range(K)]
        else:
            row = transition_c.get(hat_k)
            if row is None:
                raise KeyError(f"transition_c has no row for hat class '{hat_k}'")
            if len(row) != K:
                raise ValueError(f"transition_c['{hat_k}'] length {len(row)} != K={K}")
            c_row = list(map(float, row))

        # blend: mu * p_curr + (1-mu) * c(hat_k, k)
        p_blend = {classes[i]: mu * p_curr[classes[i]] + (1.0 - mu) * c_row[i] for i in range(K)}

        # renormalize (numerical safety)
        s = sum(max(0.0, v) for v in p_blend.values())
        if s <= 0:
            p_curr2 = p_curr
        else:
            p_curr2 = {c: max(0.0, p_blend[c]) / s for c in classes}

        p_curr = p_curr2

    # Return either probs or counts based on parameter
    if return_counts:
        return _round_probs_to_counts(p_curr, total=N)
    else:
        return p_curr


def get_tranistion_matrix(classes, map_id_counts: Dataset) -> Dict[str, List[float]]:
    K = len(classes)

    # Map class -> index
    idx = {c: i for i, c in enumerate(classes)}

    # Accumulate raw counts per majority class
    row_counts: Dict[str, List[int]] = {c: [0] * K for c in classes}

    for _id, counts in map_id_counts.items():
        if not counts:
            continue

        # majority class (argmax count)
        majority_class = max(counts, key=counts.get)

        # add raw counts into that row
        for c, n in counts.items():
            row_counts[majority_class][idx[c]] += n


    # Convert accumulated counts into probabilities
    transition_c: Dict[str, List[float]] = {}

    for cls in classes:
        total = sum(row_counts[cls])
        if total == 0:
            # fallback: uniform (or leave empty)
            transition_c[cls] = [1.0 / K] * K
        else:
            transition_c[cls] = [
                row_counts[cls][i] / total
                for i in range(K)
            ]

    return transition_c