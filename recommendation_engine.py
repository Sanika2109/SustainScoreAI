"""
recommendation_engine.py
========================
Data-driven ESG Improvement Recommendation System.

Generates actionable, pillar-specific recommendations based on
a company's ESG scores relative to the industry benchmark.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  SEVERITY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def _classify_severity(company_val: float, industry_val: float, std: float) -> str:
    """
    Classifies severity based on how far below/above the benchmark.
    Returns: 'critical', 'watch', 'strong', 'excellent'
    """
    if std == 0:
        std = 1e-8
    z = (company_val - industry_val) / std

    if z < -1.0:
        return "critical"
    elif z < -0.3:
        return "watch"
    elif z < 0.5:
        return "strong"
    else:
        return "excellent"


# ═══════════════════════════════════════════════════════════════════
#  RECOMMENDATION KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════

_E_RECOMMENDATIONS = {
    "critical": [
        "🔴 **Immediate carbon emissions audit** — Commission a third-party GHG Protocol Scope 1-3 emissions assessment.",
        "🔴 **Set Science-Based Targets** — Align with the SBTi framework for net-zero by 2050.",
        "🔴 **Supply chain sustainability review** — Mandate environmental due diligence for Tier 1 suppliers.",
        "🔴 **Increase environmental disclosure** — Adopt TCFD or SASB reporting frameworks immediately.",
    ],
    "watch": [
        "🟡 **Accelerate renewable energy adoption** — Target 50%+ renewable energy in operations within 2 years.",
        "🟡 **Implement circular economy pilots** — Reduce waste-to-landfill by 30% through recycling and reuse programs.",
        "🟡 **Enhance water stewardship** — Set context-based water targets in high-stress regions.",
        "🟡 **Publish annual sustainability report** — Include CDP Climate questionnaire scores.",
    ],
    "strong": [
        "🟢 **Maintain strong environmental practices** — Continue current sustainability programs.",
        "🟢 **Pursue leadership certifications** — Aim for CDP A-List, LEED Platinum, or ISO 14001.",
        "🟢 **Invest in biodiversity initiatives** — Support TNFD-aligned nature-positive strategies.",
    ],
    "excellent": [
        "✨ **Share best practices** — Publish environmental case studies and lead industry coalitions.",
        "✨ **Pioneer innovation** — Invest in climate tech R&D and green fintech solutions.",
    ],
}

_S_RECOMMENDATIONS = {
    "critical": [
        "🔴 **Conduct workforce equity audit** — Immediately assess pay gaps across gender, race, and tenure.",
        "🔴 **Implement whistleblower protection** — Establish independent, anonymous reporting channels.",
        "🔴 **Strengthen employee safety protocols** — Aim for zero-incident workplace within 12 months.",
        "🔴 **Launch community impact programs** — Allocate minimum 1% of revenue to CSR initiatives.",
    ],
    "watch": [
        "🟡 **Expand diversity & inclusion programs** — Set measurable DEI targets for leadership (40%+ diversity by 2028).",
        "🟡 **Improve employee welfare** — Introduce mental health support, flexible working, and skills development.",
        "🟡 **Audit supply chain labor practices** — Ensure compliance with ILO core conventions.",
        "🟡 **Enhance customer data privacy** — Adopt privacy-by-design principles and GDPR-level standards globally.",
    ],
    "strong": [
        "🟢 **Maintain social excellence** — Continue current DEI and employee engagement programs.",
        "🟢 **Expand community partnerships** — Increase investment in STEM education and local hiring.",
        "🟢 **Pursue B-Corp certification** — Validate social commitment through independent assessment.",
    ],
    "excellent": [
        "✨ **Lead industry benchmarks** — Establish social impact measurement frameworks for peers.",
        "✨ **Scale impact investing** — Direct corporate venture capital toward social enterprises.",
    ],
}

_G_RECOMMENDATIONS = {
    "critical": [
        "🔴 **Overhaul board composition** — Ensure ≥40% independent directors with ESG expertise.",
        "🔴 **Implement anti-corruption framework** — Adopt ISO 37001 anti-bribery management system.",
        "🔴 **Establish ESG committee** — Create dedicated board-level sustainability oversight.",
        "🔴 **Enhance executive compensation transparency** — Link 30%+ of variable pay to ESG KPIs.",
    ],
    "watch": [
        "🟡 **Strengthen cybersecurity governance** — Implement NIST CSF or ISO 27001 frameworks.",
        "🟡 **Improve regulatory compliance monitoring** — Deploy automated compliance tracking systems.",
        "🟡 **Enhance shareholder engagement** — Hold annual ESG-focused investor roadshows.",
        "🟡 **Publish political activity disclosures** — Ensure full transparency on lobbying expenditures.",
    ],
    "strong": [
        "🟢 **Maintain governance leadership** — Continue strong board oversight and transparency.",
        "🟢 **Refine risk management** — Integrate ESG risks into enterprise risk management (ERM).",
        "🟢 **Strengthen data ethics** — Adopt responsible AI governance frameworks.",
    ],
    "excellent": [
        "✨ **Pioneer governance innovation** — Pilot blockchain-based voting or real-time ESG reporting.",
        "✨ **Mentor other organizations** — Share governance frameworks through industry associations.",
    ],
}

_OVERALL_RECOMMENDATIONS = {
    "critical": [
        "🔴 **Urgent ESG transformation needed** — Appoint a Chief Sustainability Officer (CSO) and allocate dedicated ESG budget.",
        "🔴 **Engage ESG rating agencies** — Proactively address concerns from MSCI, Sustainalytics, and ISS ESG.",
        "🔴 **Develop 3-year ESG roadmap** — Define concrete milestones across all three pillars with board accountability.",
    ],
    "watch": [
        "🟡 **Accelerate ESG integration** — Embed sustainability into corporate strategy, not just compliance.",
        "🟡 **Improve ESG data infrastructure** — Invest in automated ESG data collection and reporting platforms.",
        "🟡 **Benchmark against leaders** — Study and adopt best practices from top-ranked peers.",
    ],
    "strong": [
        "🟢 **Continue on trajectory** — Maintain current ESG momentum while seeking areas for incremental improvement.",
        "🟢 **Explore ESG-linked financing** — Issue sustainability-linked bonds to fund green projects.",
    ],
    "excellent": [
        "✨ **World-class ESG profile** — Maintain leadership position and drive industry transformation.",
        "✨ **Influence policy** — Engage with regulators to shape future ESG disclosure standards.",
    ],
}


# ═══════════════════════════════════════════════════════════════════
#  MAIN RECOMMENDATION GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_recommendations(
    company_scores: dict,
    industry_scores: dict,
    industry_std: dict = None,
) -> dict:
    """
    Generates structured, data-driven recommendations for a company.
    Includes both raw Z-scores and scaled 0-100 display scores.
    """
    from esg_engine import to_display_score
    
    if industry_std is None:
        industry_std = {"e_mean": 0.3, "s_mean": 0.3, "g_mean": 0.3, "overall_mean": 0.3}

    pillars = [
        ("environmental", "e_mean", _E_RECOMMENDATIONS, "e_display"),
        ("social", "s_mean", _S_RECOMMENDATIONS, "s_display"),
        ("governance", "g_mean", _G_RECOMMENDATIONS, "g_display"),
        ("overall", "overall_mean", _OVERALL_RECOMMENDATIONS, "overall_display"),
    ]

    severity_icons = {
        "critical": "🔴",
        "watch": "🟡",
        "strong": "🟢",
        "excellent": "✨",
    }

    severity_labels = {
        "critical": "Critical — Significant Improvement Needed",
        "watch": "Watch — Below Average, Actionable Improvements Available",
        "strong": "Strong — Above Average Performance",
        "excellent": "Excellent — Industry Leading",
    }

    result = {}

    for pillar_name, key, rec_db, display_key in pillars:
        comp_val = company_scores.get(key, 0)
        ind_val = industry_scores.get(key, 0)
        std_val = industry_std.get(key, 0.3)

        severity = _classify_severity(comp_val, ind_val, std_val)
        gap = comp_val - ind_val
        
        # Scaling to 0-100
        comp_display = to_display_score(comp_val)
        ind_display = industry_scores.get(display_key, to_display_score(ind_val))
        gap_pts = comp_display - ind_display

        result[pillar_name] = {
            "severity": severity,
            "severity_label": severity_labels[severity],
            "icon": severity_icons[severity],
            "recommendations": rec_db.get(severity, []),
            "gap": round(gap, 4),
            "gap_pct": f"{gap / (abs(ind_val) + 1e-8) * 100:+.1f}%",
            "gap_pts": round(gap_pts, 2),
            "company_score": round(comp_val, 4),
            "industry_score": round(ind_val, 4),
            "company_display": round(comp_display, 1),
            "industry_display": round(ind_display, 1),
        }

    # Identify weakest pillar
    pillar_gaps = {
        "Environmental": company_scores.get("e_mean", 0) - industry_scores.get("e_mean", 0),
        "Social": company_scores.get("s_mean", 0) - industry_scores.get("s_mean", 0),
        "Governance": company_scores.get("g_mean", 0) - industry_scores.get("g_mean", 0),
    }

    weakest = min(pillar_gaps, key=pillar_gaps.get)
    strongest = max(pillar_gaps, key=pillar_gaps.get)

    result["summary"] = {
        "weakest_pillar": weakest,
        "strongest_pillar": strongest,
        "priority_message": (
            f"**Priority Focus Area: {weakest}** — This pillar shows the largest gap "
            f"({pillar_gaps[weakest]:+.3f}) relative to the industry benchmark. "
            f"Concentrating improvement efforts here will yield the highest ESG uplift."
        ),
        "strength_message": (
            f"**Key Strength: {strongest}** — This pillar outperforms or is closest to "
            f"the industry average ({pillar_gaps[strongest]:+.3f}). "
            f"Leverage this strength in stakeholder communications."
        ),
    }

    return result


# ═══════════════════════════════════════════════════════════════════
#  INDUSTRY STD CALCULATOR (helper for main app)
# ═══════════════════════════════════════════════════════════════════

def compute_industry_std(df_all) -> dict:
    """
    Computes per-company mean scores, then returns the std across companies.
    Used for severity classification.
    """
    from esg_engine import ensure_columns, compute_pillar_scores

    df_all = ensure_columns(df_all)
    df_all = compute_pillar_scores(df_all)

    company_means = df_all.groupby("Organization").agg(
        e_mean=("E_score", "mean"),
        s_mean=("S_score", "mean"),
        g_mean=("G_score", "mean"),
        overall_mean=("Overall_ESG", "mean"),
    )

    return {
        "e_mean": float(company_means["e_mean"].std()),
        "s_mean": float(company_means["s_mean"].std()),
        "g_mean": float(company_means["g_mean"].std()),
        "overall_mean": float(company_means["overall_mean"].std()),
    }
