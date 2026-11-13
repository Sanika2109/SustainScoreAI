from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from datetime import datetime
import io
import pandas as pd

def generate_esg_pdf(report_data):
    """
    Generates an exhaustive 1:1 Dashboard-to-PDF ESG Intelligence Report.
    """
    company_name = report_data.get("company_name", "Unknown")
    exec_metrics = report_data.get("exec_metrics", {})
    pillar_scores = report_data.get("pillar_scores", {})
    forecast_summary = report_data.get("forecast_summary", (0, "stable"))
    risks = report_data.get("risks", [])
    story = report_data.get("story", [])
    recommendations = report_data.get("recommendations", [])
    factors = report_data.get("factors", {})
    rank = report_data.get("rank", "N/A")
    total_ranked = report_data.get("total_ranked", "N/A")
    correlations = report_data.get("correlations", {})
    top_persons = report_data.get("top_persons", {})
    top_locations = report_data.get("top_locations", {})
    top_publishers = report_data.get("top_publishers", {})
    network_xai = report_data.get("network_xai", [])
    
    top_pos = report_data.get("top_pos_articles", [])
    top_neg = report_data.get("top_neg_articles", [])
    monthly_table = report_data.get("monthly_table", [])
    weekly_table = report_data.get("weekly_table", [])
    trend_slopes = report_data.get("trend_slopes", {})
    peer_comp = report_data.get("peer_comp", [])
    stats_dist = report_data.get("stats_dist", {})
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('MainTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#4F46E5"), spaceAfter=20, alignment=1)
    subtitle_style = ParagraphStyle('SubTitle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor("#1F2937"), spaceAfter=12, spaceBefore=12)
    subsubtitle_style = ParagraphStyle('SubSubTitle', parent=styles['Heading3'], fontSize=12, textColor=colors.HexColor("#374151"), spaceAfter=8)
    normal_style = styles['Normal']
    normal_style.spaceAfter = 6
    
    content = []
    
    # ═════════ TAB 1: EXECUTIVE DASHBOARD ═════════
    content.append(Paragraph(f"Exhaustive ESG Intelligence Report", title_style))
    content.append(Paragraph(f"Detailed Dashboard Review for **{company_name.title()}**", ParagraphStyle('SubTitle2', parent=styles['Heading2'], alignment=1)))
    content.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", ParagraphStyle('Date', parent=styles['Normal'], alignment=1)))
    content.append(Spacer(1, 24))

    content.append(Paragraph("Tab 1: Overview & Executive KPIs", subtitle_style))
    rank_text = f" Ranked **{rank}** out of {total_ranked} peers globally." if rank != "N/A" else ""
    sum_text = f"""The composite ESG score stands at <b>{exec_metrics.get('composite_esg', 0):.1f}</b>, reflecting a <b>{exec_metrics.get('gap', 0):+.1f} pts</b> performance gap relative to the industry benchmark.{rank_text}"""
    content.append(Paragraph(sum_text, normal_style))
    content.append(Spacer(1, 12))

    data = [
        ['Metric', 'Score', 'Context'],
        ['Composite ESG', f"{exec_metrics.get('composite_esg', 0):.1f}", "Scaled 0-100"],
        ['Industry Baseline', f"{exec_metrics.get('industry_esg', 0):.1f}", "Sector Average"],
        ['Momentum', f"{exec_metrics.get('momentum', 0):+.3f}", "Score Change Rate"],
        ['Volatility', f"{exec_metrics.get('volatility', 0):.3f}", "Sentiment Stability"]
    ]
    t = Table(data, colWidths=[150, 100, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    content.append(t)
    
    p_data = [
        ['Environmental (E)', f"{pillar_scores.get('e_mean', 0):.3f}"],
        ['Social (S)', f"{pillar_scores.get('s_mean', 0):.3f}"],
        ['Governance (G)', f"{pillar_scores.get('g_mean', 0):.3f}"]
    ]
    t2 = Table(p_data, colWidths=[200, 150])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#10B981")),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    content.append(Spacer(1, 12))
    content.append(t2)
    content.append(PageBreak())

    # ═════════ TAB 2: INDIVIDUAL SCORES ═════════
    content.append(Paragraph("Tab 2: Individual Scores & Media Highlights", subtitle_style))
    content.append(Paragraph("Most Positive Intelligence:", subsubtitle_style))
    if top_pos:
        for p in top_pos:
            content.append(Paragraph(f"• <b>Pillar:</b> {p['Dominant_Pillar']} | <b>Tone:</b> {p['Tone']:.2f}<br/><b>URL:</b> {p['URL']}", normal_style))
    else:
        content.append(Paragraph("No positive articles available.", normal_style))

    content.append(Spacer(1, 12))
    content.append(Paragraph("Most Negative Intelligence:", subsubtitle_style))
    if top_neg:
        for p in top_neg:
            content.append(Paragraph(f"• <b>Pillar:</b> {p['Dominant_Pillar']} | <b>Tone:</b> {p['Tone']:.2f}<br/><b>URL:</b> {p['URL']}", normal_style))
    else:
        content.append(Paragraph("No negative articles available.", normal_style))
    
    content.append(Spacer(1, 24))

    # ═════════ TAB 3, 4, 5: TIME SERIES ANALYSIS ═════════
    content.append(Paragraph("Tabs 3, 4, 5: Trends, Monthly & Weekly Analysis", subtitle_style))
    content.append(Paragraph("Pillar Momentum (Slope):", subsubtitle_style))
    e_slope = trend_slopes.get("E", 0)
    s_slope = trend_slopes.get("S", 0)
    g_slope = trend_slopes.get("G", 0)
    content.append(Paragraph(f"Environmental Momentum: {e_slope:+.4f}", normal_style))
    content.append(Paragraph(f"Social Momentum: {s_slope:+.4f}", normal_style))
    content.append(Paragraph(f"Governance Momentum: {g_slope:+.4f}", normal_style))
    content.append(Spacer(1, 12))
    
    content.append(Paragraph("Recent Monthly Aggregates (Top 6):", subsubtitle_style))
    if monthly_table:
        m_data = [['Month', 'Overall ESG', 'E Score', 'S Score', 'G Score']]
        for m in monthly_table:
            date_str = m['DATE'].strftime('%Y-%m') if isinstance(m['DATE'], datetime) else str(m['DATE'])[:7]
            m_data.append([date_str, f"{m.get('Overall_ESG',0):.3f}", f"{m.get('E_score',0):.3f}", f"{m.get('S_score',0):.3f}", f"{m.get('G_score',0):.3f}"])
        tm = Table(m_data)
        tm.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        content.append(tm)
    content.append(PageBreak())

    # ═════════ TAB 6 & 7: PREDICTION & RISKS ═════════
    content.append(Paragraph("Tab 6: AI Prediction", subtitle_style))
    pred_text = f"The composite score is forecasted to be <b>{forecast_summary[1]}</b> over the next period with an estimated <b>{forecast_summary[0]:+.1f}%</b> trajectory change."
    content.append(Paragraph(pred_text, normal_style))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Tab 7: Risk Intelligence", subtitle_style))
    if risks:
        for risk in risks:
            p_risk = f"• <font color='red'><b>{risk['severity']}</b></font>: {risk['message']}"
            content.append(Paragraph(p_risk, normal_style))
    else:
        content.append(Paragraph("No significant structural risks detected at this time.", normal_style))
    content.append(PageBreak())

    # ═════════ TAB 8 & 9: SCORE BREAKDOWN & STORY ═════════
    content.append(Paragraph("Tab 8: Score Breakdown (Factors)", subtitle_style))
    if factors:
        for pillar, fac in factors.items():
            content.append(Paragraph(f"<b>{pillar}:</b> {fac.get('primary_factor', 'Unknown')} - {fac.get('explanation', '')}", normal_style))
    else:
        content.append(Paragraph("Score breakdown factors not available.", normal_style))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Tab 9: Performance Story", subtitle_style))
    if story:
        for point in story:
            ev = f"• <b>{point['DATE'].strftime('%Y-%m-%d')}</b>: {point['Event']} - {point['Description']}"
            content.append(Paragraph(ev, normal_style))
    else:
        content.append(Paragraph("Consistent ESG performance monitored across all reporting channels.", normal_style))
    content.append(PageBreak())

    # ═════════ TAB 10 & 11: PEERS & DISTRIBUTIONS ═════════
    content.append(Paragraph("Tab 10: Rankings & Peer Comparison", subtitle_style))
    if peer_comp:
        pc_data = [['Organization', 'Overall ESG', 'E', 'S', 'G']]
        for pc in peer_comp:
            pc_data.append([pc.get('Organization',''), f"{pc.get('Overall_ESG',0):.1f}", f"{pc.get('E_score',0):.1f}", f"{pc.get('S_score',0):.1f}", f"{pc.get('G_score',0):.1f}"])
        tpc = Table(pc_data, colWidths=[150, 70, 70, 70, 70])
        tpc.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#3B82F6")), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        content.append(tpc)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Tab 11: Distributions & Outliers", subtitle_style))
    if stats_dist:
        metrics = list(stats_dist.keys())
        stat_rows = [['Statistic'] + metrics]
        for stat_name in ['mean', 'std', 'min', '50%', 'max']:
            row = [stat_name]
            for m in metrics:
                row.append(str(stats_dist[m].get(stat_name, '')))
            stat_rows.append(row)
        t_stat = Table(stat_rows)
        t_stat.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
        content.append(t_stat)
    content.append(PageBreak())

    # ═════════ TAB 12 & 13: CORRELATIONS & RECOMMENDATIONS ═════════
    content.append(Paragraph("Tab 12: Correlations Matrix", subtitle_style))
    if correlations and 'E_score' in correlations:
        corr_data = [
            ['', 'E Score', 'S Score', 'G Score'],
            ['E Score', str(correlations['E_score'].get('E_score', 1.0)), str(correlations['E_score'].get('S_score', '-')), str(correlations['E_score'].get('G_score', '-'))],
            ['S Score', str(correlations['S_score'].get('E_score', '-')), str(correlations['S_score'].get('S_score', 1.0)), str(correlations['S_score'].get('G_score', '-'))],
            ['G Score', str(correlations['G_score'].get('E_score', '-')), str(correlations['G_score'].get('S_score', '-')), str(correlations['G_score'].get('G_score', 1.0))]
        ]
        t_corr = Table(corr_data)
        t_corr.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#6366F1")), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('BACKGROUND', (0,1), (0,-1), colors.HexColor("#6366F1")), ('TEXTCOLOR', (0,1), (0,-1), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 1, colors.grey)]))
        content.append(t_corr)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Tab 13: Recommendations", subtitle_style))
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            content.append(Paragraph(f"{i}. {rec}", normal_style))
    content.append(PageBreak())

    # ═════════ TAB 14 & 15: ADVANCED GDELT & NETWORK XAI ═════════
    content.append(Paragraph("Tab 14: Advanced GDELT", subtitle_style))
    col_w = [140, 140, 140]
    gdelt_data = [['Top Persons', 'Top Locations', 'Top Publishers']]
    persons, locs, pubs = list(top_persons.items()), list(top_locations.items()), list(top_publishers.items())
    max_len = max(len(persons), len(locs), len(pubs))
    for i in range(max_len):
        p_str = f"{persons[i][0]} ({persons[i][1]})" if i < len(persons) else ""
        l_str = f"{locs[i][0]} ({locs[i][1]})" if i < len(locs) else ""
        pb_str = f"{pubs[i][0]} ({pubs[i][1]})" if i < len(pubs) else ""
        gdelt_data.append([p_str, l_str, pb_str])
    if max_len > 0:
        t_gdelt = Table(gdelt_data, colWidths=col_w)
        t_gdelt.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#3B82F6")), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 1, colors.grey), ('FONTSIZE', (0,1), (-1,-1), 9)]))
        content.append(t_gdelt)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Tab 15: Network XAI (Node2Vec)", subtitle_style))
    if network_xai:
        for nxai in network_xai:
            n_text = f"<b>Connected Peer:</b> {nxai['neighbor']} (Confidence: {nxai['confidence']:.2f})<br/><b>XAI Explanation:</b> {nxai['explanation']}"
            content.append(Paragraph(n_text, normal_style))
    else:
        content.append(Paragraph("No Node2Vec graph connections found for this entity.", normal_style))

    # Finalization
    doc.build(content)
    buffer.seek(0)
    return buffer
