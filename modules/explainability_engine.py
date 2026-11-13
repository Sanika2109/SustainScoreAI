

def get_score_breakdown(pillar_score, column="Overall_ESG"):
    """
    Returns a dictionary of factor contributions based on industry defaults.
    Ensures that the total weight sums to 100%.
    """
    pillar_map = {
        "E_score": {
            "Carbon Footprint": 0.40,
            "Resource Efficiency": 0.30,
            "Waste Management": 0.30,
        },
        "S_score": {
            "Labor Standards": 0.35,
            "Community Relations": 0.45,
            "Data Privacy": 0.20,
        },
        "G_score": {
            "Board Diversity": 0.40,
            "Executive Pay": 0.30,
            "Ethics & Integrity": 0.30,
        },
        "Overall_ESG": {
            "Environmental Pillar": 0.40,
            "Social Pillar": 0.30,
            "Governance Pillar": 0.30,
        }
    }

    if column not in pillar_map:
        return {}

    factors = pillar_map[column]
    breakdown = {factor: weight * pillar_score for factor, weight in factors.items()}

    return breakdown


def get_factor_percentages(column="Overall_ESG", scores=None):
    """
    Returns normalized percentage distribution of factors for a given pillar.

    For 'Overall_ESG', dynamically computes the proportional share of each
    pillar using its absolute raw score:
        pct_i = |score_i| / (|score_E| + |score_S| + |score_G|) × 100
    This guarantees the pillar with the highest individual score always shows
    the highest percentage, and the display is consistent with the Individual
    Scores tab (e.g. G=0.698 correctly ranks above E=0.628).

    For individual pillars (E_score, S_score, G_score), fixed industry-standard
    sub-factor weights are returned (no per-row scores available).

    Parameters
    ----------
    column : str
        The selected pillar key ('Overall_ESG', 'E_score', 'S_score', 'G_score').
    scores : dict, optional
        company_summary dict containing 'e_mean', 's_mean', 'g_mean'.
        Only used when column == 'Overall_ESG'.

    Returns
    -------
    list of (str, float) tuples  [(factor_name, percentage), ...]
    """
    # ── Dynamic computation for Overall_ESG ──────────────────────────
    if column == "Overall_ESG" and scores is not None:
        e_mean = scores.get("e_mean", 0)
        s_mean = scores.get("s_mean", 0)
        g_mean = scores.get("g_mean", 0)

        # Use absolute raw scores so the pillar with the highest individual
        # score always shows the highest percentage (e.g. G=0.698 > E=0.628).
        abs_e = abs(e_mean)
        abs_s = abs(s_mean)
        abs_g = abs(g_mean)

        total = abs_e + abs_s + abs_g

        if total < 1e-9:
            # All scores are essentially zero — fall back to static weights
            return [("Environmental", 40.0), ("Social", 30.0), ("Governance", 30.0)]

        pct_e = round(abs_e / total * 100, 1)
        pct_s = round(abs_s / total * 100, 1)
        # Assign remainder to G to guarantee the three values sum to exactly 100
        pct_g = round(100 - pct_e - pct_s, 1)

        return [("Environmental", pct_e), ("Social", pct_s), ("Governance", pct_g)]

    # ── Static weights for individual pillars ────────────────────────
    # For individual pillars we derive approximate signal-level magnitudes
    # from the available pillar means using the pillar formula weights:
    #   G = tone×0.7 + articles×0.3  →  tone_z   ≈ g_mean / 0.7
    #   E = articles×0.6 + tone×0.4  →  articles_z ≈ (e_mean − 0.4×tone_z) / 0.6
    #   S = mentions×0.5 + tone×0.5  →  mentions_z ≈ (s_mean − 0.5×tone_z) / 0.5
    if scores is not None and column in ("E_score", "S_score", "G_score"):
        e_mean = scores.get("e_mean", 0)
        s_mean = scores.get("s_mean", 0)
        g_mean = scores.get("g_mean", 0)

        # Derive approximate global-normalised signal means
        tone_z     = g_mean / 0.7 if abs(g_mean) > 1e-9 else 0.0
        articles_z = (e_mean - 0.4 * tone_z) / 0.6 if abs(e_mean) > 1e-9 else 0.0
        mentions_z = (s_mean - 0.5 * tone_z) / 0.5 if abs(s_mean) > 1e-9 else 0.0

        abs_t = abs(tone_z)
        abs_a = abs(articles_z)
        abs_m = abs(mentions_z)

        def _normalise_pcts(vals):
            total = sum(vals)
            if total < 1e-9:
                return [round(100 / len(vals), 1)] * len(vals)
            pcts = [round(v / total * 100, 1) for v in vals]
            # Ensure exact sum = 100 by adjusting the last element
            pcts[-1] = round(100 - sum(pcts[:-1]), 1)
            return pcts

        if column == "E_score":
            # E = articles×0.6 + tone×0.4
            # Carbon Footprint  ← articles component (volume = emissions coverage)
            # Resource Efficiency ← half of tone component
            # Waste Management    ← half of tone component
            cf  = abs_a * 0.6
            re_ = abs_t * 0.4 * 0.5
            wm  = abs_t * 0.4 * 0.5
            pcts = _normalise_pcts([cf, re_, wm])
            return [("Carbon Footprint", pcts[0]),
                    ("Resource Efficiency", pcts[1]),
                    ("Waste Management", pcts[2])]

        if column == "S_score":
            # S = mentions×0.5 + tone×0.5
            # Labor Standards    ← 70% of mentions component
            # Community Relations← 30% of mentions + 60% of tone component
            # Data Privacy        ← 40% of tone component
            ls  = abs_m * 0.5 * 0.70
            cr  = abs_m * 0.5 * 0.30 + abs_t * 0.5 * 0.60
            dp  = abs_t * 0.5 * 0.40
            pcts = _normalise_pcts([ls, cr, dp])
            return [("Labor Standards", pcts[0]),
                    ("Community Relations", pcts[1]),
                    ("Data Privacy", pcts[2])]

        if column == "G_score":
            # G = tone×0.7 + articles×0.3
            # Board Diversity    ← articles component (diversity press coverage)
            # Executive Pay      ← 43% of tone component
            # Ethics & Integrity ← 57% of tone component
            bd  = abs_a * 0.3
            ep  = abs_t * 0.7 * 0.43
            ei  = abs_t * 0.7 * 0.57
            pcts = _normalise_pcts([bd, ep, ei])
            return [("Board Diversity", pcts[0]),
                    ("Executive Pay", pcts[1]),
                    ("Ethics & Integrity", pcts[2])]

    # Fallback: static weights when no scores are provided
    static_map = {
        "E_score": [("Carbon Footprint", 40), ("Resource Efficiency", 30), ("Waste Management", 30)],
        "S_score": [("Labor Standards", 35), ("Community Relations", 45), ("Data Privacy", 20)],
        "G_score": [("Board Diversity", 40), ("Executive Pay", 30), ("Business Ethics", 30)],
        "Overall_ESG": [("Environmental", 40), ("Social", 30), ("Governance", 30)],
    }
    return static_map.get(column, [])



