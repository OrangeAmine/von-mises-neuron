"""Build a PowerPoint presentation comparing VonMises vs LSTM only.

Reads figures from results/presentation_figures_vm_lstm/ and assembles them
into a focused .pptx for the head-to-head VM vs LSTM comparison.

Usage:
    python scripts/build_presentation_vm_lstm.py
    python scripts/build_presentation_vm_lstm.py --figures-dir results/presentation_figures_vm_lstm
    python scripts/build_presentation_vm_lstm.py --output results/vm_vs_lstm.pptx
"""

import csv
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    scale_w = max_width / img_w
    scale_h = max_height / img_h
    scale = min(scale_w, scale_h)
    final_w = int(img_w * scale)
    final_h = int(img_h * scale)
    left = (SLIDE_WIDTH - Emu(final_w * 914400 // 96)) // 2
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
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)

    add_textbox(slide, Inches(1), Inches(1.8), Inches(11.3), Inches(1.5),
                "Von Mises Neuron vs LSTM", font_size=44, bold=True,
                color=WHITE, alignment=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(1), Inches(3.2), Inches(11.3), Inches(1),
                "Head-to-Head: Periodicity-Preserving vs Standard Architecture",
                font_size=24, color=ACCENT_BLUE, alignment=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(1), Inches(4.8), Inches(11.3), Inches(0.6),
                "s = exp(\u03ba \u00b7 cos(\u03b8 + \u03bc))",
                font_size=28, color=ACCENT_ORANGE,
                alignment=PP_ALIGN.CENTER, font_name="Consolas")

    add_textbox(slide, Inches(1), Inches(6.0), Inches(11.3), Inches(0.5),
                "Lab Meeting | 2026",
                font_size=16, color=MEDIUM_GRAY, alignment=PP_ALIGN.CENTER)

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "This version focuses on the direct comparison between VonMises and LSTM. "
        "Both models have ~43k parameters. The key question: does the VM encoder's "
        "periodicity-preserving property give a measurable advantage over standard LSTM?"
    )


def slide_problem(prs):
    """Slide 2: The problem."""
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

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "LEFT: ReLU(w*theta + b) grows without bound - no periodicity. "
        "RIGHT: exp(kappa*cos(theta+mu)) repeats perfectly every 2pi. "
        "Moving the weight outside the cosine preserves the period."
    )


def slide_architecture(prs, figures_dir):
    """Slide 3: Architecture comparison."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8),
                "Architecture Comparison",
                font_size=32, bold=True, color=DARK_TEXT)

    # Left: VonMises
    tf = add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(4.5),
                     "VonMisesForecaster", font_size=22, bold=True, color=ACCENT_BLUE)
    add_bullet_text(tf, "", font_size=4)
    add_bullet_text(tf, "1. VM Encoder (per-timestep)", font_size=17, bold=True)
    add_bullet_text(tf, "[VonMisesLinear \u2192 LayerNorm \u2192 Dropout] \u00d7 N",
                    font_size=14, font_name="Consolas", level=1)
    add_bullet_text(tf, "Extracts periodic features, 2\u03c0-equivariant", font_size=14, level=1)
    add_bullet_text(tf, "", font_size=4)
    add_bullet_text(tf, "2. LSTM Temporal Backbone", font_size=17, bold=True)
    add_bullet_text(tf, "Captures temporal dynamics", font_size=14, level=1)
    add_bullet_text(tf, "", font_size=4)
    add_bullet_text(tf, "3. Linear Output Head", font_size=17, bold=True)
    add_bullet_text(tf, "Multi-step forecast", font_size=14, level=1)

    # Right: LSTM
    tf2 = add_textbox(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(4.5),
                      "LSTM Baseline", font_size=22, bold=True, color=ACCENT_ORANGE)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "1. Linear+ReLU Encoder", font_size=17, bold=True)
    add_bullet_text(tf2, "[Linear \u2192 ReLU \u2192 LayerNorm \u2192 Dropout] \u00d7 N",
                    font_size=14, font_name="Consolas", level=1)
    add_bullet_text(tf2, "Standard feature extraction", font_size=14, level=1)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "2. LSTM Temporal Backbone", font_size=17, bold=True)
    add_bullet_text(tf2, "Same architecture as VM", font_size=14, level=1)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "3. Linear Output Head", font_size=17, bold=True)
    add_bullet_text(tf2, "Same architecture as VM", font_size=14, level=1)
    add_bullet_text(tf2, "", font_size=8)
    add_bullet_text(tf2, "Parameter-matched (~43k params each)", font_size=15,
                    color=MEDIUM_GRAY, bold=True)
    add_bullet_text(tf2, "Only difference: VM encoder vs Linear+ReLU encoder",
                    font_size=15, color=MEDIUM_GRAY)

    # Architecture diagram at bottom
    img_path = figures_dir / "01_architecture_diagram.png"
    if img_path.exists():
        add_image_centered(slide, img_path, Inches(5.0),
                           max_width=Inches(10).inches * 96,
                           max_height=Inches(2.2).inches * 96)

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "The two models are identical except for the encoder. "
        "VM uses exp(kappa*cos(theta+mu)), LSTM uses Linear+ReLU. "
        "Same LSTM backbone, same output head, same parameter count. "
        "This isolates the effect of the periodicity-preserving encoder."
    )


def slide_setup(prs):
    """Slide 4: Experimental setup."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8),
                "Experimental Setup",
                font_size=32, bold=True, color=DARK_TEXT)

    # Left: Fair comparison
    tf = add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5),
                     "Fair comparison:", font_size=20, bold=True, color=DARK_TEXT)
    add_bullet_text(tf, "", font_size=4)
    add_bullet_text(tf, "Both models: ~43,000 parameters", font_size=17, bold=True)
    add_bullet_text(tf, "Same LSTM backbone + linear head", font_size=15, level=1)
    add_bullet_text(tf, "Only encoder differs (VM vs Linear+ReLU)", font_size=15, level=1)
    add_bullet_text(tf, "", font_size=6)
    add_bullet_text(tf, "Same training recipe:", font_size=17, bold=True)
    add_bullet_text(tf, "Adam, LR=0.001, ReduceLROnPlateau", font_size=15, level=1)
    add_bullet_text(tf, "Early stopping (patience=15)", font_size=15, level=1)
    add_bullet_text(tf, "Gradient clipping = 1.0", font_size=15, level=1)
    add_bullet_text(tf, "Same random seed (42)", font_size=15, level=1)
    add_bullet_text(tf, "", font_size=6)
    add_bullet_text(tf, "Metrics on denormalized scale:", font_size=17, bold=True)
    add_bullet_text(tf, "RMSE, MAE, R\u00b2, MAPE", font_size=15, level=1)

    # Right: Datasets
    tf2 = add_textbox(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(5),
                      "Datasets:", font_size=20, bold=True, color=DARK_TEXT)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "Synthetic CO2", font_size=17, bold=True)
    add_bullet_text(tf2, "600 + 50sin(daily) + 15sin(weekly) + noise", font_size=13,
                    level=1, font_name="Consolas")
    add_bullet_text(tf2, "90 days, lookback=72h, horizon=12h", font_size=13, level=1,
                    color=MEDIUM_GRAY)
    add_bullet_text(tf2, "Known periodic structure \u2192 VM should excel", font_size=13,
                    level=1, color=ACCENT_BLUE)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "UCI Occupancy (CO2)", font_size=17, bold=True)
    add_bullet_text(tf2, "346 hourly rows, lookback=24h, horizon=6h", font_size=13,
                    level=1, color=MEDIUM_GRAY)
    add_bullet_text(tf2, "Small dataset stress test", font_size=13, level=1)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "Jena Climate (Temperature)", font_size=17, bold=True)
    add_bullet_text(tf2, "70k hourly rows, lookback=168h, horizon=24h", font_size=13,
                    level=1, color=MEDIUM_GRAY)
    add_bullet_text(tf2, "Diurnal + seasonal cycles", font_size=13, level=1)

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "Parameter matching is critical for a fair comparison. "
        "Both models see the exact same data, with the same splits and preprocessing. "
        "The only variable is the encoder architecture."
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
    """Slide with a summary metrics table."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_textbox(slide, Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.7),
                "Head-to-Head Results", font_size=32, bold=True, color=DARK_TEXT,
                alignment=PP_ALIGN.CENTER)

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

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

    n_cols = 6  # Model, Params, RMSE, MAE, R2, MAPE
    n_rows = 1
    for ds_name in datasets:
        n_rows += 1 + len(datasets[ds_name])

    table_width = Inches(11)
    table_left = (SLIDE_WIDTH - table_width) // 2
    table_top = Inches(1.3)
    table_height = Inches(5.5)

    table_shape = slide.shapes.add_table(
        n_rows, n_cols, table_left, table_top, table_width, table_height
    )
    table = table_shape.table

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

                if row_idx % 2 == 0:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(0xF5, 0xF5, 0xF5)

            row_idx += 1

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "Bold green = best RMSE per dataset. Direct head-to-head comparison "
        "with identical parameter counts. The only architectural difference is "
        "the VM encoder vs Linear+ReLU encoder."
    )


def slide_takeaways(prs):
    """Final slide: key takeaways."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BG)

    add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8),
                "Key Takeaways", font_size=32, bold=True, color=WHITE)

    tf = add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5),
                     "", font_size=18, color=WHITE)
    add_bullet_text(tf, "VM neuron preserves 2\u03c0-periodicity exactly",
                    font_size=18, color=WHITE, bold=True)
    add_bullet_text(tf, "f(\u03b8) = f(\u03b8 + 2\u03c0) by construction",
                    font_size=14, color=LIGHT_GRAY, level=1)
    add_bullet_text(tf, "", font_size=8)
    add_bullet_text(tf, "Outperforms LSTM on periodic synthetic data",
                    font_size=18, color=WHITE, bold=True)
    add_bullet_text(tf, "Lower RMSE with same parameter budget",
                    font_size=14, color=LIGHT_GRAY, level=1)
    add_bullet_text(tf, "", font_size=8)
    add_bullet_text(tf, "Competitive on real-world data",
                    font_size=18, color=WHITE, bold=True)
    add_bullet_text(tf, "Matches LSTM on Jena Climate (diurnal + seasonal)",
                    font_size=14, color=LIGHT_GRAY, level=1)
    add_bullet_text(tf, "", font_size=8)
    add_bullet_text(tf, "Interpretable learned parameters",
                    font_size=18, color=WHITE, bold=True)
    add_bullet_text(tf, "\u03ba reveals which features matter, \u03bc shows preferred phase",
                    font_size=14, color=LIGHT_GRAY, level=1)

    tf2 = add_textbox(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(5),
                      "The case for VM over LSTM:", font_size=20, bold=True, color=ACCENT_BLUE)
    add_bullet_text(tf2, "", font_size=8)
    add_bullet_text(tf2, "Same performance, stronger inductive bias",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "Periodicity preserved without learning it",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "Interpretable: inspect learned phases",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "No manual sin/cos feature engineering",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "", font_size=12)
    add_bullet_text(tf2, "Future work:", font_size=20, bold=True, color=ACCENT_ORANGE)
    add_bullet_text(tf2, "", font_size=4)
    add_bullet_text(tf2, "Circular domains (wind, protein dihedrals)",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "VM + attention mechanisms",
                    font_size=16, color=LIGHT_GRAY)
    add_bullet_text(tf2, "Multi-scale periodicity detection",
                    font_size=16, color=LIGHT_GRAY)

    notes = slide.notes_slide
    notes.notes_text_frame.text = (
        "Main pitch: VM gets you the same or better performance as LSTM, "
        "with the bonus of built-in periodicity preservation and interpretability. "
        "The VM encoder is a drop-in replacement for Linear+ReLU."
    )


def build_presentation(figures_dir: Path, output_path: Path):
    """Assemble the VM vs LSTM presentation."""
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT

    # --- Intro slides ---
    slide_title(prs)
    slide_problem(prs)
    slide_architecture(prs, figures_dir)
    slide_setup(prs)

    # --- Results: Synthetic ---
    slide_section_header(prs, "Results: Synthetic CO2",
                         "Controlled benchmark with known periodic structure")

    slide_figure(prs, "Predictions: VonMises vs LSTM -- Synthetic CO2",
                 figures_dir / "02a_predictions_synthetic.png",
                 notes_text="Both models track the signal well. VM is closer to ground truth "
                 "on the periodic components, as expected from its built-in periodicity bias.",
                 subtitle="Test set predictions (100-sample window)")

    slide_figure(prs, "Metrics: VonMises vs LSTM -- Synthetic CO2",
                 figures_dir / "03a_metrics_synthetic.png",
                 notes_text="VM beats LSTM on all metrics. The periodicity-preserving encoder "
                 "gives a clear advantage on data with known periodic structure.")

    slide_figure(prs, "Training Dynamics -- Synthetic CO2",
                 figures_dir / "05a_training_curves_synthetic.png",
                 notes_text="Both models converge at similar rates. "
                 "VM reaches a lower final loss, consistent with its metric advantage.")

    # --- Results: Occupancy ---
    slide_section_header(prs, "Results: UCI Occupancy",
                         "Small dataset stress test (346 hourly rows)")

    slide_figure(prs, "Predictions: VonMises vs LSTM -- UCI Occupancy",
                 figures_dir / "02b_predictions_occupancy.png",
                 notes_text="Both models struggle with this tiny dataset. "
                 "Only 213 training sequences. CO2 spikes are hard to predict.",
                 subtitle="Very small test set (23 samples)")

    slide_figure(prs, "Metrics: VonMises vs LSTM -- UCI Occupancy",
                 figures_dir / "03b_metrics_occupancy.png",
                 notes_text="Negative R2 means both models are worse than the mean. "
                 "Expected with so little data. Neither model wins clearly.")

    # --- Results: Jena Climate ---
    slide_section_header(prs, "Results: Jena Climate",
                         "Real-world temperature forecasting (70k rows, 8 years)")

    slide_figure(prs, "Predictions: VonMises vs LSTM -- Jena Climate",
                 figures_dir / "02c_predictions_jena.png",
                 notes_text="Both models capture the diurnal temperature cycle. "
                 "VM and LSTM produce nearly identical predictions on this data.",
                 subtitle="Day-ahead temperature forecast (test window)")

    slide_figure(prs, "Metrics: VonMises vs LSTM -- Jena Climate",
                 figures_dir / "03c_metrics_jena.png",
                 notes_text="Very close performance. VM is competitive with LSTM "
                 "while providing interpretability benefits.",
                 subtitle="RMSE in degrees Celsius")

    slide_figure(prs, "Training Dynamics -- Jena Climate",
                 figures_dir / "05c_training_curves_jena.png",
                 notes_text="Both models converge well. VM has stable training "
                 "dynamics -- no exploding gradients despite exp() activation.")

    # --- Interpretability (VM advantage) ---
    slide_section_header(prs, "Interpretability",
                         "What did the VM neurons learn? (LSTM has no equivalent)")

    slide_figure(prs, "VM Neuron Response Curves -- Synthetic CO2",
                 figures_dir / "04a_vm_response_synthetic.png",
                 notes_text="Each curve is one learned neuron. Different neurons tune to "
                 "different phases and sharpnesses, tiling the full angular range. "
                 "LSTM weights have no such interpretable structure.",
                 subtitle="Top-8 neurons by concentration (first VM layer)")

    slide_figure(prs, "VM Neuron Response Curves -- Jena Climate",
                 figures_dir / "04c_vm_response_jena.png",
                 notes_text="On real data, neurons develop different specializations. "
                 "Sharp neurons detect specific phases, broad ones give general response. "
                 "This interpretability is a unique advantage over LSTM.")

    slide_figure(prs, "Learned Phase & Concentration -- Synthetic CO2",
                 figures_dir / "06a_phase_kappa_synthetic.png",
                 notes_text="Phase (mu) and concentration (kappa) heatmaps. "
                 "Neurons spread phases across [0, 2pi]. Kappa shows feature selectivity.",
                 subtitle="Heatmaps of learned \u03bc (phase) and \u03ba (concentration)")

    slide_figure(prs, "Learned Phase & Concentration -- Jena Climate",
                 figures_dir / "06c_phase_kappa_jena.png",
                 notes_text="On real data, kappa varies across features. "
                 "Wind direction and temperature get higher kappa -- the model "
                 "learns feature-specific periodicity.",
                 subtitle="Layer 1 parameters after training")

    # --- Summary ---
    slide_metrics_table(prs, figures_dir / "metrics_summary.csv")
    slide_takeaways(prs)

    prs.save(str(output_path))
    print(f"Presentation saved to: {output_path}")
    print(f"Total slides: {len(prs.slides)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build VM vs LSTM PowerPoint presentation"
    )
    parser.add_argument(
        "--figures-dir", type=str,
        default="results/presentation_figures_vm_lstm",
        help="Directory containing generated figures",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/von_mises_vs_lstm.pptx",
        help="Output .pptx path",
    )
    args = parser.parse_args()

    figures_dir = PROJECT_ROOT / args.figures_dir
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_presentation(figures_dir, output_path)


if __name__ == "__main__":
    main()
