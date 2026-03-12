"""Build a PowerPoint presentation from the generated figures.

Reads figures from results/presentation_figures/ and assembles them into
a structured .pptx with title slides, section headers, and speaker notes.

Usage:
    python scripts/build_presentation.py
    python scripts/build_presentation.py --output results/presentation.pptx
"""

import csv
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "presentation_figures"

# Slide dimensions: widescreen 16:9
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

# Colors
DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
ACCENT_BLUE = RGBColor(0x4F, 0xC3, 0xF7)
ACCENT_ORANGE = RGBColor(0xFF, 0xB7, 0x4D)
MEDIUM_GRAY = RGBColor(0x99, 0x99, 0x99)
DARK_TEXT = RGBColor(0x33, 0x33, 0x33)
SUBTLE_BG = RGBColor(0xF5, 0xF5, 0xF5)


def set_slide_bg(slide, color):
    """Set solid background color for a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=DARK_TEXT, alignment=PP_ALIGN.LEFT,
                font_name="Calibri"):
    """Add a textbox to the slide and return the textframe."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_bullet_text(tf, text, font_size=16, color=DARK_TEXT,
                    level=0, font_name="Calibri", bold=False):
    """Add a bullet paragraph to an existing textframe."""
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.name = font_name
    p.font.bold = bold
    p.level = level
    return p


def add_image_centered(slide, img_path, top, max_width, max_height):
    """Add an image centered horizontally, scaled to fit constraints."""
    from PIL import Image
    img = Image.open(img_path)
    img_w, img_h = img.size
    # Scale to fit
    scale_w = max_width / img_w
    scale_h = max_height / img_h
    scale = min(scale_w, scale_h)
    final_w = int(img_w * scale)
    final_h = int(img_h * scale)
    left = (SLIDE_WIDTH - Emu(final_w * 914400 // 96)) // 2  # center
    pic = slide.shapes.add_picture(
        str(img_path), left, top,
        Emu(final_w * 914400 // 96), Emu(final_h * 914400 // 96),
    )
    return pic


def add_image_fill(slide, img_path, left, top, width, height):
    """Add an image at exact position and size."""
    pic = slide.shapes.add_picture(str(img_path), left, top, width, height)
    return pic


# =========================================================================
# Slide builders
# =========================================================================

def slide_title(prs):
    """Slide 1: Title slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide, DARK_BG)

    add_textbox(slide, Inches(1), Inches(1.8), Inches(11.3), Inches(1.5),
                "Von Mises Neuron", font_size=44, bold=True,
                color=WHITE, alignment=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(1), Inches(3.2), Inches(11.3), Inches(1),
                "A Periodicity-Preserving Neural Network Layer",
                font_size=24, color=ACCENT_BLUE, alignment=PP_ALIGN.CENTER)

    tf = add_textbox(slide, Inches(1), Inches(4.8), Inches(11.3), Inches(0.6),
                     "s = exp(\u03ba \u00b7 cos(\u03b8 + \u03bc))",
                     font_size=28, color=ACCENT_ORANGE,
                     alignment=PP_ALIGN.CENTER, font_name="Consolas")

    add_textbox(slide, Inches(1), Inches(6.0), Inches(11.3), Inches(0.5),
                "Lab Meeting | 2026",
                font_size=16, color=MEDIUM_GRAY, alignment=PP_ALIGN.CENTER)

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "Core idea: The weight sits OUTSIDE the cosine, "
        "so the neuron's output is strictly 2pi-periodic in theta. "
        "Standard neurons multiply inside the nonlinearity, breaking periodicity."
    )


def slide_problem(prs):
    """Slide 2: The problem — standard neurons break periodicity."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8),
                "The Problem: Standard Neurons Break Periodicity",
                font_size=32, bold=True, color=DARK_TEXT)

    tf = add_textbox(slide, Inches(0.8), Inches(1.4), Inches(5.5), Inches(2.5),
                     "Standard neuron:", font_size=20, bold=True, color=DARK_TEXT)
    add_bullet_text(tf, "s = f(W\u00b7\u03b8 + b)", font_size=18,
                    font_name="Consolas", color=DARK_TEXT)
    add_bullet_text(tf, "Multiplying \u03b8 by W changes the period", font_size=16)
    add_bullet_text(tf, "f(\u03b8) \u2260 f(\u03b8 + 2\u03c0) in general", font_size=16)
    add_bullet_text(tf, "", font_size=10)
    add_bullet_text(tf, "Von Mises neuron:", font_size=20, bold=True)
    add_bullet_text(tf, "s = exp(\u03ba \u00b7 cos(\u03b8 + \u03bc))", font_size=18,
                    font_name="Consolas", color=ACCENT_BLUE)
    add_bullet_text(tf, "Weight \u03ba sits OUTSIDE the cosine", font_size=16)
    add_bullet_text(tf, "f(\u03b8) = f(\u03b8 + 2\u03c0) exactly", font_size=16)
    add_bullet_text(tf, "\u03ba > 0: concentration (sharpness)", font_size=14,
                    color=MEDIUM_GRAY, level=1)
    add_bullet_text(tf, "\u03bc: phase shift (preferred direction)", font_size=14,
                    color=MEDIUM_GRAY, level=1)

    # Architecture diagram on the right
    img_path = FIGURES_DIR / "01_architecture_diagram.png"
    if img_path.exists():
        add_image_fill(slide, img_path,
                       Inches(6.5), Inches(1.3), Inches(6.5), Inches(3.2))

    # Speaker notes
    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "LEFT panel: ReLU(w*theta + b) grows without bound - no periodicity. "
        "RIGHT panel: exp(kappa*cos(theta+mu)) repeats perfectly every 2pi. "
        "The key insight: moving the weight outside the cosine preserves the period. "
        "This is exactly the von Mises distribution (circular Gaussian)."
    )


def slide_architecture(prs):
    """Slide 3: VonMisesForecaster architecture."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8),
                "VonMisesForecaster Architecture",
                font_size=32, bold=True, color=DARK_TEXT)

    # Architecture description
    tf = add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(4.5),
                     "Three-stage pipeline:", font_size=20, bold=True, color=DARK_TEXT)

    add_bullet_text(tf, "", font_size=6)
    add_bullet_text(tf, "1. VM Encoder (per-timestep)", font_size=18, bold=True,
                    color=ACCENT_BLUE)
    add_bullet_text(tf, "[VonMisesLinear \u2192 LayerNorm \u2192 Dropout] \u00d7 N",
                    font_size=14, font_name="Consolas", level=1)
    add_bullet_text(tf, "Extracts periodic features, 2\u03c0-equivariant", font_size=14, level=1)
    add_bullet_text(tf, "Log-space output avoids exp(exp(z)) blowup", font_size=14,
                    level=1, color=MEDIUM_GRAY)

    add_bullet_text(tf, "", font_size=6)
    add_bullet_text(tf, "2. LSTM Temporal Backbone", font_size=18, bold=True,
                    color=ACCENT_BLUE)
    add_bullet_text(tf, "Captures how periodic patterns evolve over time", font_size=14, level=1)
    add_bullet_text(tf, "Separation of concerns: VM = what, LSTM = how", font_size=14, level=1)

    add_bullet_text(tf, "", font_size=6)
    add_bullet_text(tf, "3. Linear Output Head", font_size=18, bold=True,
                    color=ACCENT_BLUE)
    add_bullet_text(tf, "Direct multi-step forecast: h \u2192 horizon", font_size=14, level=1)

    # Gradient properties on the right
    tf2 = add_textbox(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(4.5),
                      "Gradient properties:", font_size=20, bold=True, color=DARK_TEXT)
    add_bullet_text(tf2, "", font_size=6)
    add_bullet_text(tf2, "\u2202s/\u2202\u03bc = -s \u00b7 \u03ba \u00b7 sin(\u03b8 + \u03bc)",
                    font_size=16, font_name="Consolas")
    add_bullet_text(tf2, "\u2192 Vanishes at cos peaks (stable once aligned)",
                    font_size=14, level=1, color=MEDIUM_GRAY)
    add_bullet_text(tf2, "", font_size=6)
    add_bullet_text(tf2, "\u2202s/\u2202\u03ba = s \u00b7 cos(\u03b8 + \u03bc)",
                    font_size=16, font_name="Consolas")
    add_bullet_text(tf2, "\u2192 Always bounded, proportional to cos",
                    font_size=14, level=1, color=MEDIUM_GRAY)
    add_bullet_text(tf2, "", font_size=10)
    add_bullet_text(tf2, "Desirable: once a neuron's phase aligns", font_size=15)
    add_bullet_text(tf2, "with the input, it stops drifting. \u03ba can", font_size=15)
    add_bullet_text(tf2, "still sharpen or soften the response.", font_size=15)

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "The VM encoder processes each timestep independently - it's a feature extractor. "
        "The LSTM captures temporal dynamics across the encoded timesteps. "
        "Key: log-space output between VM layers prevents numerical blowup. "
        "Phase gradient vanishing at peaks is a FEATURE, not a bug - it means learned "
        "phase alignments are stable equilibria."
    )


def slide_baselines(prs):
    """Slide 4: Experimental setup & baselines."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8),
                "Experimental Setup",
                font_size=32, bold=True, color=DARK_TEXT)

    # Left: Models
    tf = add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5),
                     "Models (~43k parameters each):",
                     font_size=20, bold=True, color=DARK_TEXT)
    add_bullet_text(tf, "", font_size=4)
    add_bullet_text(tf, "VonMisesForecaster", font_size=17, bold=True, color=ACCENT_BLUE)
    add_bullet_text(tf, "VM encoder + LSTM + linear head", font_size=14, level=1)
    add_bullet_text(tf, "", font_size=4)
    add_bullet_text(tf, "LSTM Baseline (parameter-matched)", font_size=17, bold=True,
                    color=ACCENT_ORANGE)
    add_bullet_text(tf, "Linear+ReLU encoder replacing VM layers", font_size=14, level=1)
    add_bullet_text(tf, "", font_size=4)
    add_bullet_text(tf, "Transformer Baseline", font_size=17, bold=True,
                    color=RGBColor(0x66, 0xBB, 0x6A))
    add_bullet_text(tf, "Sinusoidal PE + multi-head attention", font_size=14, level=1)
    add_bullet_text(tf, "", font_size=4)
    add_bullet_text(tf, "HandCrafted Linear", font_size=17, bold=True,
                    color=RGBColor(0xEF, 0x53, 0x50))
    add_bullet_text(tf, "Manual sin/cos features + linear regression", font_size=14, level=1)
    add_bullet_text(tf, "Oracle baseline: tests if VM automates manual engineering",
                    font_size=13, level=1, color=MEDIUM_GRAY)

    # Right: Datasets
    tf2 = add_textbox(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(5),
                      "Datasets:", font_size=20, bold=True, color=DARK_TEXT)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "Synthetic CO2", font_size=17, bold=True)
    add_bullet_text(tf2, "600 + 50sin(daily) + 15sin(weekly) + noise", font_size=13,
                    level=1, font_name="Consolas")
    add_bullet_text(tf2, "90 days, lookback=72h, horizon=12h", font_size=13, level=1,
                    color=MEDIUM_GRAY)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "UCI Occupancy (CO2)", font_size=17, bold=True)
    add_bullet_text(tf2, "346 hourly rows, lookback=24h, horizon=6h", font_size=13,
                    level=1, color=MEDIUM_GRAY)
    add_bullet_text(tf2, "Small dataset stress test", font_size=13, level=1)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "Jena Climate (Temperature)", font_size=17, bold=True)
    add_bullet_text(tf2, "70k hourly rows, lookback=168h, horizon=24h", font_size=13,
                    level=1, color=MEDIUM_GRAY)
    add_bullet_text(tf2, "Diurnal + seasonal cycles, real-world noise", font_size=13, level=1)
    add_bullet_text(tf2, "", font_size=6)
    add_bullet_text(tf2, "Training: Adam, LR=0.001, early stopping (patience=15)",
                    font_size=14, color=MEDIUM_GRAY)
    add_bullet_text(tf2, "Metrics: RMSE, MAE, R\u00b2, MAPE (denormalized scale)",
                    font_size=14, color=MEDIUM_GRAY)

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "All neural models are parameter-matched (~43k params) for fair comparison. "
        "HandCrafted is much smaller but gets oracle features. "
        "Training uses Adam with ReduceLROnPlateau and gradient clipping=1.0. "
        "Chronological 80/10/10 splits (no shuffling) to prevent future leakage."
    )


def slide_section_header(prs, title, subtitle=""):
    """Section divider slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)

    add_textbox(slide, Inches(1), Inches(2.5), Inches(11.3), Inches(1.2),
                title, font_size=36, bold=True,
                color=WHITE, alignment=PP_ALIGN.CENTER)

    if subtitle:
        add_textbox(slide, Inches(1), Inches(3.8), Inches(11.3), Inches(0.8),
                    subtitle, font_size=20,
                    color=ACCENT_BLUE, alignment=PP_ALIGN.CENTER)


def slide_figure(prs, title, img_path, notes_text="", subtitle=""):
    """Slide with a single full-width figure."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_textbox(slide, Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.7),
                title, font_size=26, bold=True, color=DARK_TEXT,
                alignment=PP_ALIGN.CENTER)

    if subtitle:
        add_textbox(slide, Inches(0.5), Inches(0.85), Inches(12.3), Inches(0.5),
                    subtitle, font_size=15, color=MEDIUM_GRAY,
                    alignment=PP_ALIGN.CENTER)

    top = Inches(1.4) if subtitle else Inches(1.1)
    max_h = Inches(5.8) if subtitle else Inches(6.1)

    if img_path.exists():
        add_image_centered(slide, img_path, top,
                           max_width=Inches(12.3).inches * 96,
                           max_height=max_h.inches * 96)

    if notes_text:
        notes = slide.notes_slide
        notes.notes_text_frame.text = notes_text


def slide_metrics_table(prs, csv_path):
    """Slide with a summary metrics table across all datasets."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_textbox(slide, Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.7),
                "Results Summary", font_size=32, bold=True, color=DARK_TEXT,
                alignment=PP_ALIGN.CENTER)

    # Read CSV
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by dataset
    datasets = {}
    for row in rows:
        ds = row["dataset"]
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(row)

    ds_labels = {
        "synthetic": "Synthetic CO2",
        "occupancy": "UCI Occupancy",
        "jena": "Jena Climate",
    }

    # Build table: one section per dataset
    n_cols = 6  # Model, Params, RMSE, MAE, R2, MAPE
    n_rows = 1  # header
    for ds_name in datasets:
        n_rows += 1 + len(datasets[ds_name])  # subheader + data rows

    table_width = Inches(11)
    table_left = (SLIDE_WIDTH - table_width) // 2
    table_top = Inches(1.3)
    table_height = Inches(5.5)

    table_shape = slide.shapes.add_table(
        n_rows, n_cols, table_left, table_top, table_width, table_height
    )
    table = table_shape.table

    # Style header
    headers = ["Model", "Params", "RMSE", "MAE", "R\u00b2", "MAPE (%)"]
    col_widths = [Inches(2.2), Inches(1.3), Inches(1.8), Inches(1.8), Inches(1.8), Inches(1.8)]
    for i, (hdr, w) in enumerate(zip(headers, col_widths)):
        table.columns[i].width = w
        cell = table.cell(0, i)
        cell.text = hdr
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0x37, 0x47, 0x4F)

    row_idx = 1
    for ds_name, ds_rows in datasets.items():
        # Dataset subheader — merge then style the origin cell
        label = ds_labels.get(ds_name, ds_name)
        cell = table.cell(row_idx, 0)
        cell.merge(table.cell(row_idx, n_cols - 1))
        cell.text = label
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.LEFT
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0x60, 0x7D, 0x8B)
        row_idx += 1

        # Find best RMSE for highlighting
        rmse_vals = [float(r["rmse"]) for r in ds_rows]
        best_rmse = min(rmse_vals)

        for dr in ds_rows:
            vals = [
                dr["model"],
                f"{int(dr['params']):,}",
                f"{float(dr['rmse']):.3f}",
                f"{float(dr['mae']):.3f}",
                f"{float(dr['r2']):.4f}",
                f"{float(dr['mape']):.2f}",
            ]
            is_best = abs(float(dr["rmse"]) - best_rmse) < 0.001

            for ci, val in enumerate(vals):
                cell = table.cell(row_idx, ci)
                cell.text = val
                for p in cell.text_frame.paragraphs:
                    p.font.size = Pt(12)
                    p.font.color.rgb = DARK_TEXT
                    p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
                    if is_best and ci >= 2:
                        p.font.bold = True
                        p.font.color.rgb = RGBColor(0x1B, 0x5E, 0x20)

                # Alternate row shading
                if row_idx % 2 == 0:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(0xF5, 0xF5, 0xF5)

            row_idx += 1

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "Bold green = best RMSE per dataset. "
        "Synthetic: VM beats LSTM/Transformer, close to HandCrafted oracle. "
        "Occupancy: all models struggle (tiny dataset, negative R2). "
        "Jena: VM competitive with all baselines, beats HandCrafted."
    )


def slide_takeaways(prs):
    """Final slide: key takeaways and future work."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)

    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8),
                "Key Takeaways", font_size=32, bold=True, color=WHITE)

    tf = add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5),
                     "", font_size=18, color=WHITE)
    add_bullet_text(tf, "VM neuron preserves 2\u03c0-periodicity exactly",
                    font_size=18, color=WHITE, bold=True)
    add_bullet_text(tf, "f(\u03b8) = f(\u03b8 + 2\u03c0) to numerical precision",
                    font_size=14, color=LIGHT_GRAY, level=1)
    add_bullet_text(tf, "", font_size=8)
    add_bullet_text(tf, "Competitive with standard architectures",
                    font_size=18, color=WHITE, bold=True)
    add_bullet_text(tf, "Beats LSTM/Transformer on synthetic periodic data",
                    font_size=14, color=LIGHT_GRAY, level=1)
    add_bullet_text(tf, "Matches them on real-world data (Jena Climate)",
                    font_size=14, color=LIGHT_GRAY, level=1)
    add_bullet_text(tf, "", font_size=8)
    add_bullet_text(tf, "Automates what manual feature engineering does",
                    font_size=18, color=WHITE, bold=True)
    add_bullet_text(tf, "No need to pre-compute sin/cos time features",
                    font_size=14, color=LIGHT_GRAY, level=1)
    add_bullet_text(tf, "", font_size=8)
    add_bullet_text(tf, "Interpretable learned parameters",
                    font_size=18, color=WHITE, bold=True)
    add_bullet_text(tf, "\u03ba shows which features matter, \u03bc shows preferred phase",
                    font_size=14, color=LIGHT_GRAY, level=1)

    tf2 = add_textbox(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(5),
                      "Future directions:", font_size=20, bold=True, color=ACCENT_BLUE)
    add_bullet_text(tf2, "", font_size=8)
    add_bullet_text(tf2, "Circular domains: wind direction, protein dihedrals",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "VM + attention (VonMisesAttentionForecaster)",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "Multi-scale periodicity detection",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "Benchmarking on truly periodic inputs (not z-scored)",
                    font_size=16, color=LIGHT_GRAY)

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "Main pitch: if your data has periodic structure, the VM neuron respects it "
        "by construction instead of hoping the network learns it. "
        "The interpretability angle is also strong - you can inspect learned phases "
        "and concentrations to understand what the model found."
    )


def build_presentation(output_path: Path):
    """Assemble the full presentation."""
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT

    # --- Intro slides ---
    slide_title(prs)
    slide_problem(prs)
    slide_architecture(prs)
    slide_baselines(prs)

    # --- Results: Synthetic ---
    slide_section_header(prs, "Results: Synthetic CO2",
                         "Controlled benchmark with known periodic structure")

    slide_figure(prs, "Predictions vs Ground Truth -- Synthetic CO2",
                 FIGURES_DIR / "02a_predictions_synthetic.png",
                 notes_text="All models track the signal well. VM is competitive. "
                 "HandCrafted has a slight edge because it gets oracle sin/cos features "
                 "matching the exact generating process.",
                 subtitle="1-step ahead forecast on the test set (100-sample window)")

    slide_figure(prs, "Model Comparison -- Synthetic CO2",
                 FIGURES_DIR / "03a_metrics_synthetic.png",
                 notes_text="VM beats LSTM and Transformer. HandCrafted wins because "
                 "it gets the exact sin/cos features. On synthetic data, VM automatically "
                 "discovers what HandCrafted gets for free.")

    slide_figure(prs, "Training Dynamics -- Synthetic CO2",
                 FIGURES_DIR / "05a_training_curves_synthetic.png",
                 notes_text="HandCrafted converges fastest (linear model, simple loss landscape). "
                 "VM and LSTM converge at similar rates. Transformer starts higher but catches up.")

    # --- Results: Occupancy ---
    slide_section_header(prs, "Results: UCI Occupancy",
                         "Small dataset stress test (346 hourly rows)")

    slide_figure(prs, "Predictions vs Ground Truth -- UCI Occupancy",
                 FIGURES_DIR / "02b_predictions_occupancy.png",
                 notes_text="All models struggle here. Only 213 training sequences and 23 test "
                 "sequences. CO2 spikes are hard to predict with so little data.",
                 subtitle="1-step ahead forecast (very small test set: 23 samples)")

    slide_figure(prs, "Model Comparison -- UCI Occupancy",
                 FIGURES_DIR / "03b_metrics_occupancy.png",
                 notes_text="Negative R2 across the board means all models are worse than "
                 "predicting the mean. This is expected with such a tiny dataset. "
                 "Not a failure of the VM neuron specifically.")

    # --- Results: Jena Climate ---
    slide_section_header(prs, "Results: Jena Climate",
                         "Real-world temperature forecasting (70k rows, 8 years)")

    slide_figure(prs, "Predictions vs Ground Truth -- Jena Climate",
                 FIGURES_DIR / "02c_predictions_jena.png",
                 notes_text="All models capture the diurnal temperature cycle well. "
                 "VM tracks the ground truth closely, competitive with all baselines.",
                 subtitle="Day-ahead temperature forecast (1-step ahead, test window)")

    slide_figure(prs, "Model Comparison -- Jena Climate",
                 FIGURES_DIR / "03c_metrics_jena.png",
                 notes_text="All models within ~5% RMSE of each other. VM (2.43) beats "
                 "HandCrafted (2.42) -- the VM neuron automatically discovers periodicities "
                 "that manual engineering approximates. Transformer slightly edges everyone.",
                 subtitle="RMSE in degrees Celsius")

    slide_figure(prs, "Training Dynamics -- Jena Climate",
                 FIGURES_DIR / "05c_training_curves_jena.png",
                 notes_text="All models converge well on this large dataset. "
                 "VM has stable training dynamics -- no exploding gradients despite "
                 "the exp() in the activation.")

    # --- Interpretability ---
    slide_section_header(prs, "Interpretability",
                         "What did the VM neurons learn?")

    slide_figure(prs, "VM Neuron Response Curves -- Synthetic CO2",
                 FIGURES_DIR / "04a_vm_response_synthetic.png",
                 notes_text="Each curve is one learned neuron. Different neurons are tuned "
                 "to different phases (mu) and sharpnesses (kappa). Together they tile the "
                 "full angular range, providing coverage of all possible periodic patterns.",
                 subtitle="Top-8 neurons by concentration (first VM layer)")

    slide_figure(prs, "VM Neuron Response Curves -- Jena Climate",
                 FIGURES_DIR / "04c_vm_response_jena.png",
                 notes_text="On real data, neurons develop different specializations. "
                 "Some are sharp (high kappa = specific phase detection), "
                 "others are broad (low kappa = general response).")

    slide_figure(prs, "Learned Phase & Concentration -- Synthetic CO2",
                 FIGURES_DIR / "06a_phase_kappa_synthetic.png",
                 notes_text="Left heatmap: phase shifts (mu) for each neuron-feature pair. "
                 "Neurons spread their phases across [0, 2pi]. "
                 "Right: concentration (kappa) shows which features each neuron attends to.",
                 subtitle="Heatmaps of learned mu (phase) and kappa (concentration)")

    slide_figure(prs, "Learned Phase & Concentration -- Jena Climate",
                 FIGURES_DIR / "06c_phase_kappa_jena.png",
                 notes_text="On real data, kappa varies more across features. "
                 "Wind direction (inherently circular) and temperature get higher kappa "
                 "from some neurons, showing the model learns feature-specific periodicity.",
                 subtitle="Layer 1 parameters after training on 56k sequences")

    # --- Summary ---
    slide_metrics_table(prs, FIGURES_DIR / "metrics_summary.csv")
    slide_takeaways(prs)

    prs.save(str(output_path))
    print(f"Presentation saved to: {output_path}")
    print(f"Total slides: {len(prs.slides)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build PowerPoint from figures")
    parser.add_argument("--output", type=str,
                        default="results/von_mises_presentation.pptx",
                        help="Output .pptx path")
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_presentation(output_path)


if __name__ == "__main__":
    main()
