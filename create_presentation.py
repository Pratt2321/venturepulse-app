import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

OUTPUT_PPTX = "VenturePulse_Midterm_Presentation_Deck.pptx"
OUTPUT_PDF = "VenturePulse_Midterm_Presentation_Deck.pdf"

# Color Palette
DARK_BG = RGBColor(11, 17, 33)       # Deep slate #0B1121
CARD_BG = RGBColor(22, 33, 54)       # Deep navy card #162136
BORDER_COLOR = RGBColor(56, 189, 248) # Cyan border #38BDF8
WHITE = RGBColor(255, 255, 255)
MUTED_TEXT = RGBColor(148, 163, 184) # Slate gray #94A3B8
CYAN = RGBColor(0, 242, 254)         # Neon cyan #00F2FE
GREEN = RGBColor(16, 185, 129)       # Emerald green #10B981
AMBER = RGBColor(245, 158, 11)       # Amber gold #F59E0B

def create_shape(slide, shape_type, left, top, width, height, fill_color, border_color=None):
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape

def add_header(slide, title_text, category_badge="CMSE 830 • MIDTERM PROJECT READINESS"):
    create_shape(slide, MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5), DARK_BG)
    
    badge_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8), Inches(0.35))
    tf_b = badge_box.text_frame
    tf_b.word_wrap = True
    p_b = tf_b.paragraphs[0]
    p_b.text = category_badge.upper()
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = CYAN

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.65))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(21)
    p_t.font.bold = True
    p_t.font.color.rgb = WHITE

def build_presentation():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ==========================================
    # SLIDE 1: Title & Overview
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    create_shape(slide1, MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5), DARK_BG)
    
    t_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.1), Inches(11.7), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p0 = tf1.paragraphs[0]
    p0.text = "🦄 VENTUREPULSE: STARTUP VALUATION INTELLIGENCE"
    p0.font.size = Pt(28)
    p0.font.bold = True
    p0.font.color.rgb = WHITE
    
    p1 = tf1.add_paragraph()
    p1.text = "Reverse-Engineering Private Tech Pricing Across 1,447 Global Unicorns"
    p1.font.size = Pt(16)
    p1.font.color.rgb = CYAN
    p1.space_before = Pt(8)
    
    p2 = tf1.add_paragraph()
    p2.text = "Pratham Pradhan  |  CMSE 830: Foundations of Data Science  |  Fall 2026"
    p2.font.size = Pt(12)
    p2.font.color.rgb = MUTED_TEXT
    p2.space_before = Pt(8)

    metrics = [
        ("TOTAL VALUE TRACKED", "$7.40 Trillion", "Across 1,447 private tech unicorns worldwide"),
        ("DATASET COVERAGE", "55 Countries", "Spanning 6 core industries and 10+ tech hubs"),
        ("CORE OBJECTIVE", "Market Transparency", "Demystifying pricing negotiated behind closed doors")
    ]
    for i, (label, val, sub) in enumerate(metrics):
        left_pos = Inches(0.8 + i * 4.0)
        create_shape(slide1, MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(3.7), Inches(3.7), Inches(2.8), CARD_BG, BORDER_COLOR)
        c_box = slide1.shapes.add_textbox(left_pos + Inches(0.2), Inches(3.9), Inches(3.3), Inches(2.4))
        ctf = c_box.text_frame
        ctf.word_wrap = True
        
        cp1 = ctf.paragraphs[0]
        cp1.text = label
        cp1.font.size = Pt(11)
        cp1.font.bold = True
        cp1.font.color.rgb = MUTED_TEXT
        
        cp2 = ctf.add_paragraph()
        cp2.text = val
        cp2.font.size = Pt(22)
        cp2.font.bold = True
        cp2.font.color.rgb = WHITE
        cp2.space_before = Pt(8)
        
        cp3 = ctf.add_paragraph()
        cp3.text = sub
        cp3.font.size = Pt(11)
        cp3.font.color.rgb = CYAN
        cp3.space_before = Pt(10)

    # ==========================================
    # SLIDE 2: Data Engineering & Preparation
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Data Hygiene: Taming the Power-Law Distribution", "PART 2 • DATA READINESS & QUALITY")

    # Left card: Skewness & Log-Transformation
    create_shape(slide2, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3), CARD_BG, BORDER_COLOR)
    b2_left = slide2.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.3), Inches(2.5))
    tf2_l = b2_left.text_frame
    tf2_l.word_wrap = True
    
    lp0 = tf2_l.paragraphs[0]
    lp0.text = "1. THE SKEWNESS CHALLENGE"
    lp0.font.size = Pt(13)
    lp0.font.bold = True
    lp0.font.color.rgb = CYAN

    bullets_l = [
        ("Extreme Skewness (22.18):", "Valuations range from $1.0B to $965B (Anthropic). Modeling raw dollars causes linear regression to collapse with massive variance."),
        ("Log Transformation log(1 + y):", "Linearizes exponential compound returns, reducing skewness to 2.69 and stabilizing model errors.")
    ]
    for b_title, b_desc in bullets_l:
        bp1 = tf2_l.add_paragraph()
        bp1.text = f"• {b_title} "
        bp1.font.bold = True
        bp1.font.size = Pt(10.5)
        bp1.font.color.rgb = WHITE
        bp1.space_before = Pt(6)
        
        bp2 = tf2_l.add_paragraph()
        bp2.text = f"  {b_desc}"
        bp2.font.size = Pt(9.5)
        bp2.font.color.rgb = MUTED_TEXT

    if os.path.exists("slide2_skewness.png"):
        slide2.shapes.add_picture("slide2_skewness.png", Inches(1.0), Inches(4.0), width=Inches(5.3))

    # Right card: Entity Duplicates & Imputation
    create_shape(slide2, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), CARD_BG, BORDER_COLOR)
    b2_right = slide2.shapes.add_textbox(Inches(7.0), Inches(1.65), Inches(5.3), Inches(4.9))
    tf2_r = b2_right.text_frame
    tf2_r.word_wrap = True
    
    rp0 = tf2_r.paragraphs[0]
    rp0.text = "2. DUPLICATES & IMPUTATION DECISIONS"
    rp0.font.size = Pt(13)
    rp0.font.bold = True
    rp0.font.color.rgb = GREEN

    bullets_r = [
        ("The Homonym Disambiguation:", "Auditing duplicate names caught 2 copy-pasted duplicate rows (Ramp, Invoca), but preserved 3 legitimate name-collision pairs:\n  - Bolt US ($11B) vs. Bolt Estonia ($8.4B)\n  - Branch Metrics ($4B) vs. Branch Insurance ($1.05B)\n  - Fabric Washington ($1.5B) vs. Fabric NY ($1.0B)."),
        ("Informative Missingness (Option C):", "50 companies kept investor data confidential. We coded their investor count to 1 and elite backer flags to 0 rather than fabricating fake data."),
        ("Geospatial Centroid Imputation:", "23 remote startups without listed cities were assigned national centroid coordinates with random coordinate jitter to prevent map stacking.")
    ]
    for b_title, b_desc in bullets_r:
        bp1 = tf2_r.add_paragraph()
        bp1.text = f"• {b_title} "
        bp1.font.bold = True
        bp1.font.size = Pt(10.5)
        bp1.font.color.rgb = WHITE
        bp1.space_before = Pt(8)
        
        bp2 = tf2_r.add_paragraph()
        bp2.text = f"  {b_desc}"
        bp2.font.size = Pt(9.5)
        bp2.font.color.rgb = MUTED_TEXT

    # ==========================================
    # SLIDE 3: Key Finding 1 (Geography)
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Key Finding 1: The Silicon Valley Wealth Monopoly", "PART 3 • EXPLORATORY DATA ANALYSIS (QUESTION 1)")

    create_shape(slide3, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(4.5), Inches(5.3), CARD_BG, BORDER_COLOR)
    b3_l = slide3.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.1), Inches(4.9))
    tf3_l = b3_l.text_frame
    tf3_l.word_wrap = True
    
    s3_0 = tf3_l.paragraphs[0]
    s3_0.text = "GLOBAL WEALTH SHARE"
    s3_0.font.size = Pt(11)
    s3_0.font.bold = True
    s3_0.font.color.rgb = CYAN
    
    s3_1 = tf3_l.add_paragraph()
    s3_1.text = "47.5%"
    s3_1.font.size = Pt(36)
    s3_1.font.bold = True
    s3_1.font.color.rgb = WHITE
    s3_1.space_before = Pt(2)
    
    s3_2 = tf3_l.add_paragraph()
    s3_2.text = "Held by the SF Bay Area alone ($3,496 Billion out of $7,360 Billion total global unicorn capitalization)."
    s3_2.font.size = Pt(10.5)
    s3_2.font.color.rgb = MUTED_TEXT
    s3_2.space_before = Pt(4)

    s3_3 = tf3_l.add_paragraph()
    s3_3.text = "MEDIAN BASELINE DISPARITY"
    s3_3.font.size = Pt(11)
    s3_3.font.bold = True
    s3_3.font.color.rgb = GREEN
    s3_3.space_before = Pt(14)
    
    s3_4 = tf3_l.add_paragraph()
    s3_4.text = "$2.00B vs $1.60B"
    s3_4.font.size = Pt(24)
    s3_4.font.bold = True
    s3_4.font.color.rgb = WHITE
    s3_4.space_before = Pt(2)
    
    s3_5 = tf3_l.add_paragraph()
    s3_5.text = "Typical Bay Area startup commands a 25% price premium over global baseline. Average is pulled to $9.90B by mega-titans."
    s3_5.font.size = Pt(10.5)
    s3_5.font.color.rgb = MUTED_TEXT
    s3_5.space_before = Pt(4)

    create_shape(slide3, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.3), CARD_BG, BORDER_COLOR)
    b3_r = slide3.shapes.add_textbox(Inches(5.8), Inches(1.65), Inches(6.5), Inches(0.8))
    tf3_r = b3_r.text_frame
    tf3_r.word_wrap = True
    rp = tf3_r.paragraphs[0]
    rp.text = "REGIONAL ECOSYSTEM BREAKDOWN ($7.40T TOTAL)"
    rp.font.size = Pt(13)
    rp.font.bold = True
    rp.font.color.rgb = CYAN

    if os.path.exists("slide3_geography.png"):
        slide3.shapes.add_picture("slide3_geography.png", Inches(5.8), Inches(2.2), width=Inches(6.5))

    # ==========================================
    # SLIDE 4: Key Finding 2 (Frontier AI Multiplier)
    # ==========================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Key Finding 2: The Frontier AI Valuation Multiplier", "PART 3 • EXPLORATORY DATA ANALYSIS (QUESTION 2)")

    create_shape(slide4, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(4.6), Inches(5.3), CARD_BG, BORDER_COLOR)
    b4_l = slide4.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(4.2), Inches(4.9))
    tf4_l = b4_l.text_frame
    tf4_l.word_wrap = True

    s4_0 = tf4_l.paragraphs[0]
    s4_0.text = "THE 8.5X ARTIFICIAL INTELLIGENCE PREMIUM"
    s4_0.font.size = Pt(12)
    s4_0.font.bold = True
    s4_0.font.color.rgb = CYAN

    ai_bullets = [
        ("Average AI Valuation ($30.0B vs $3.5B):", "Startups developing frontier foundation models average more than 8.5 times the valuation of traditional software peers."),
        ("Enterprise Tech Dominance (54.3%):", "Enterprise Tech & AI captures $3,994 Billion—over half of all global private unicorn value."),
        ("The Centicorn Vanguard:", "The top of the pyramid is dominated by foundation model creators: Anthropic ($965B), OpenAI ($840B), Databricks ($190B), and DeepSeek ($50B).")
    ]
    for b_title, b_desc in ai_bullets:
        bp1 = tf4_l.add_paragraph()
        bp1.text = f"• {b_title}"
        bp1.font.bold = True
        bp1.font.size = Pt(10.5)
        bp1.font.color.rgb = WHITE
        bp1.space_before = Pt(8)
        
        bp2 = tf4_l.add_paragraph()
        bp2.text = f"  {b_desc}"
        bp2.font.size = Pt(9.5)
        bp2.font.color.rgb = MUTED_TEXT

    create_shape(slide4, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.7), Inches(1.5), Inches(6.8), Inches(5.3), CARD_BG, BORDER_COLOR)
    b4_r = slide4.shapes.add_textbox(Inches(5.9), Inches(1.65), Inches(6.4), Inches(0.8))
    tf4_r = b4_r.text_frame
    tf4_r.word_wrap = True

    s4_r0 = tf4_r.paragraphs[0]
    s4_r0.text = "EMPIRICAL VALUATION MULTIPLIERS & REGRESSION WEIGHTS"
    s4_r0.font.size = Pt(13)
    s4_r0.font.bold = True
    s4_r0.font.color.rgb = GREEN

    if os.path.exists("slide4_ai_multiplier.png"):
        slide4.shapes.add_picture("slide4_ai_multiplier.png", Inches(5.9), Inches(2.3), width=Inches(6.4))

    # ==========================================
    # SLIDE 5: Streamlit Product & Interaction
    # ==========================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "The Streamlit Data Product: VenturePulse Interactive Engine", "PART 5 • STREAMLIT APPLICATION & DECISION ENGINE")

    create_shape(slide5, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.2), Inches(5.3), CARD_BG, BORDER_COLOR)
    b5_l = slide5.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(4.8), Inches(4.9))
    tf5_l = b5_l.text_frame
    tf5_l.word_wrap = True

    f_p0 = tf5_l.paragraphs[0]
    f_p0.text = "PRODUCT ARCHITECTURE & CONTROLS"
    f_p0.font.size = Pt(13)
    f_p0.font.bold = True
    f_p0.font.color.rgb = CYAN

    features = [
        ("🗺️ TAB 1: 3D Geospatial Map:", "Scatter plot of 1,447 unicorns with log-scaled bubbles and perspective toggles (Natural Earth, 3D Globe, Mercator)."),
        ("🎛️ TAB 2: VC Valuation Simulator:", "Predictive pricing tool with sliders (Age, Sector, Location, Elite VC Backers). Computes estimated valuation with 80% confidence interval and finds nearest 3 peer comps."),
        ("⚡ TAB 3: Decacorn Velocity Screener:", "Identifies hyper-scaling outliers creating the highest annual enterprise value, plus full model weight transparency.")
    ]
    for f_title, f_desc in features:
        p1 = tf5_l.add_paragraph()
        p1.text = f"{f_title}"
        p1.font.bold = True
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = WHITE
        p1.space_before = Pt(8)
        
        p2 = tf5_l.add_paragraph()
        p2.text = f"{f_desc}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = MUTED_TEXT

    create_shape(slide5, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.3), Inches(1.5), Inches(6.2), Inches(5.3), CARD_BG, BORDER_COLOR)
    b5_r = slide5.shapes.add_textbox(Inches(6.5), Inches(1.65), Inches(5.8), Inches(0.6))
    tf5_r = b5_r.text_frame
    tf5_r.word_wrap = True
    p_dash = tf5_r.paragraphs[0]
    p_dash.text = "LIVE STREAMLIT DASHBOARD PREVIEW (app/app.py)"
    p_dash.font.size = Pt(12)
    p_dash.font.bold = True
    p_dash.font.color.rgb = GREEN

    if os.path.exists("streamlit_dashboard.png"):
        slide5.shapes.add_picture("streamlit_dashboard.png", Inches(6.5), Inches(2.2), width=Inches(5.8))

    # ==========================================
    # SLIDE 6: Boundaries & Final Project Roadmap
    # ==========================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Limitations & Final Project Horizon", "PART 6 • BOUNDARIES & ROADMAP")

    create_shape(slide6, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3), CARD_BG, BORDER_COLOR)
    b6_l = slide6.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(5.3), Inches(4.9))
    tf6_l = b6_l.text_frame
    tf6_l.word_wrap = True

    lp6 = tf6_l.paragraphs[0]
    lp6.text = "CURRENT PROJECT LIMITATIONS"
    lp6.font.size = Pt(13)
    lp6.font.bold = True
    lp6.font.color.rgb = AMBER

    lims = [
        ("Survivorship Bias:", "Our dataset only tracks startups that already crossed the $1B threshold. It does not predict early-stage startup bankruptcy or insolvency."),
        ("Valuation Step-Function Opacity:", "Private tech valuations only update when a new funding round is announced. A company's true operational health may drift between pricing events."),
        ("Selection vs. Causation:", "Elite VCs like Sequoia command high valuation correlations, but partly because they have privileged deal access to select the most promising founders.")
    ]
    for l_title, l_desc in lims:
        p1 = tf6_l.add_paragraph()
        p1.text = f"• {l_title}"
        p1.font.bold = True
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = WHITE
        p1.space_before = Pt(8)
        
        p2 = tf6_l.add_paragraph()
        p2.text = f"  {l_desc}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = MUTED_TEXT

    create_shape(slide6, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), CARD_BG, BORDER_COLOR)
    b6_r = slide6.shapes.add_textbox(Inches(7.0), Inches(1.65), Inches(5.3), Inches(4.9))
    tf6_r = b6_r.text_frame
    tf6_r.word_wrap = True

    rp6 = tf6_r.paragraphs[0]
    rp6.text = "ROADMAP FOR THE FINAL PROJECT"
    rp6.font.size = Pt(13)
    rp6.font.bold = True
    rp6.font.color.rgb = GREEN

    plans = [
        ("NLP & Text Analytics:", "Incorporate company pitch summaries and mission statements using text embeddings (TF-IDF or transformer embeddings) to identify semantic product trends."),
        ("Unsupervised Cluster Discovery:", "Apply K-Means or t-SNE / UMAP dimensionality reduction to discover emerging startup archetypes beyond broad industry labels."),
        ("Deployment & Cloud Hosting:", "Deploy the Streamlit application to Streamlit Community Cloud so classmates and instructors can interact on mobile/web without local terminal setup.")
    ]
    for p_title, p_desc in plans:
        p1 = tf6_r.add_paragraph()
        p1.text = f"• {p_title}"
        p1.font.bold = True
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = WHITE
        p1.space_before = Pt(8)
        
        p2 = tf6_r.add_paragraph()
        p2.text = f"  {p_desc}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = MUTED_TEXT

    prs.save(OUTPUT_PPTX)
    print(f"Presentation saved successfully to {OUTPUT_PPTX}!")

def export_pdf_deck():
    with PdfPages(OUTPUT_PDF) as pdf:
        slides = [
            ("VenturePulse: Startup Valuation Intelligence", "CMSE 830 Midterm Presentation  |  Pratham Pradhan  |  Fall 2026",
             ["• Tracks 1,447 global unicorns across 55 countries representing $7.40 Trillion in total wealth.",
              "• Investigates the extreme opacity of private tech pricing and funding dynamics.",
              "• Evaluates how much of a billion-dollar price tag is driven by geography, age, and AI hype."],
             None),
            ("Data Hygiene: Taming the Power-Law Distribution", "Addressing extreme statistical skewness and duplicate entities",
             ["• Raw valuations exhibit severe skewness (22.18), ranging from $1.0B to $965B (Anthropic).",
              "• Natural log transformation log(1 + y) reduced skewness to 2.69, stabilizing regression variance.",
              "• Auditing duplicate names identified 2 copy-pasted duplicate rows (Ramp, Invoca) while preserving legitimate homonym companies (Bolt US vs. Bolt Estonia, Branch Metrics vs. Branch Insurance).",
              "• Imputed 23 missing cities via regional centroids with jitter and coded 50 confidential investors to baseline 1."],
             "slide2_skewness.png"),
            ("Key Finding 1: The Silicon Valley Monopoly", "Geographic concentration across global tech hubs",
             ["• The SF Bay Area commands 47.5% of all global private startup capitalization ($3,496B of $7,360B).",
              "• Bay Area median valuation ($2.00B) is 25% higher than the global baseline ($1.60B).",
              "• Bay Area mean valuation reaches $9.90B, driven by clustering of mega-titans (OpenAI, Anthropic, Stripe, Databricks).",
              "• Second largest ecosystems: Other US Hubs ($1,022B, 13.9%) and China Tech Hubs ($1,001B, 13.6%)."],
             "slide3_geography.png"),
            ("Key Finding 2: The Frontier AI Multiplier", "Foundation model valuation premiums vs. traditional software",
             ["• Enterprise Tech & AI commands $3,994 Billion (54.3% of all global private unicorn value).",
              "• Frontier AI foundation model creators average $30.0 Billion—more than 8.5x the average non-AI startup ($3.5B).",
              "• Linear regression elasticity confirms Frontier AI status commands the single highest valuation premium (+78.6%), followed closely by Sequoia Capital backing (+77.3%) and a16z backing (+48.9%)."],
             "slide4_ai_multiplier.png"),
            ("The Streamlit Data Product & Decision Tool", "Interactive live demo architecture (app/app.py)",
             ["• Tab 1: Global 3D Map displaying all 1,447 unicorns with logarithmic bubble scaling and perspective toggles.",
              "• Tab 2: VC Valuation Simulator allowing real-time prediction inference with 80% confidence intervals and nearest comps.",
              "• Tab 3: Decacorn Velocity Screener identifying hyper-scaling outliers creating the highest annual enterprise value."],
             "streamlit_dashboard.png"),
            ("Project Boundaries & Final Project Horizon", "Current limitations and technical extensions",
             ["• Survivorship Bias: Dataset exclusively models winners that crossed $1B; does not predict early-stage failure.",
              "• Step-Function Pricing: Private paper valuations only adjust during discrete funding rounds, causing reporting lag.",
              "• Final Project Extension: Incorporating NLP text embeddings and unsupervised clustering on startup pitch descriptions."],
             None)
        ]
        
        for title, sub, bullets, img_path in slides:
            fig = plt.figure(figsize=(13.333, 7.5), facecolor="#0B1121")
            
            if img_path and os.path.exists(img_path):
                # Two column layout in PDF
                ax_text = fig.add_axes([0.06, 0.1, 0.45, 0.8])
                ax_img = fig.add_axes([0.54, 0.15, 0.42, 0.65])
                ax_img.set_facecolor("#0B1121")
                img = plt.imread(img_path)
                ax_img.imshow(img)
                ax_img.axis("off")
            else:
                ax_text = fig.add_axes([0.06, 0.1, 0.88, 0.8])
                
            ax_text.set_facecolor("#0B1121")
            ax_text.text(0.0, 0.96, "CMSE 830 • MIDTERM PROJECT READINESS", color="#00F2FE", fontsize=10, fontweight="bold", parse_math=False)
            ax_text.text(0.0, 0.88, title, color="white", fontsize=19, fontweight="bold", parse_math=False)
            ax_text.text(0.0, 0.81, sub, color="#94A3B8", fontsize=11, parse_math=False)
            
            import textwrap
            y_pos = 0.68
            for bullet in bullets:
                wrapped_lines = textwrap.wrap(bullet, width=58 if img_path else 110)
                formatted_text = "\n".join(wrapped_lines)
                ax_text.text(0.0, y_pos, formatted_text, color="#E2E8F0", fontsize=10.5, parse_math=False)
                y_pos -= 0.05 * len(wrapped_lines) + 0.06
                
            ax_text.text(0.0, 0.02, "Pratham Pradhan  |  Fall 2026  |  GitHub: Pratt2321/venturepulse-app", color="#64748B", fontsize=9, parse_math=False)
            ax_text.axis("off")
            
            pdf.savefig(fig, facecolor="#0B1121")
            plt.close()
    print(f"PDF deck saved successfully to {OUTPUT_PDF}!")

if __name__ == "__main__":
    build_presentation()
    export_pdf_deck()
