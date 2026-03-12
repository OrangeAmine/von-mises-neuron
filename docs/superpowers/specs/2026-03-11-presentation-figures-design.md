# Presentation Figure Generator — Design Spec

## Goal

Create a Python script that trains 4 models (VonMises, LSTM, Transformer, HandCrafted) on 4 datasets (synthetic + occupancy + power + jena), then generates 6 categories of publication-quality figures for a lab presentation.

## Figure Categories

1. **Architecture diagram** — VM neuron response vs standard neuron response over θ ∈ [0, 4π]
2. **Prediction overlays** — 4 models' predictions vs ground truth on a test window (×4 datasets)
3. **Metric bar charts** — RMSE/MAE/R²/MAPE grouped by model (×4 datasets)
4. **VM response curves** — top 8 learned neurons plotted as exp(κ·cos(θ+μ)) (×4 datasets)
5. **Training curves** — train+val loss over epochs for all 4 models (×4 datasets)
6. **Phase/concentration heatmaps** — learned μ and κ matrices (×4 datasets)

## Technical Approach

- Single script: `scripts/generate_presentation_figures.py`
- Reuses existing infrastructure: data loading, model classes, normalization, sequences, splits
- Training loop modified from existing `_train_model()` to also record per-epoch losses
- Training loop also returns the trained model (not just metrics) for interpretability extraction
- All figures saved at 300 DPI as PNG to `results/presentation_figures/`
- A `metrics_summary.csv` with all numeric results
- Seaborn default style, no custom theming
- Headless matplotlib backend (`Agg`) for compatibility

## Output Structure

```
results/presentation_figures/
  01_architecture_diagram.png
  02a_predictions_synthetic.png
  02b_predictions_occupancy.png
  02c_predictions_power.png
  02d_predictions_jena.png
  03a_metrics_synthetic.png
  03b_metrics_occupancy.png
  03c_metrics_power.png
  03d_metrics_jena.png
  04a_vm_response_synthetic.png
  04b_vm_response_occupancy.png
  04c_vm_response_power.png
  04d_vm_response_jena.png
  05a_training_curves_synthetic.png
  05b_training_curves_occupancy.png
  05c_training_curves_power.png
  05d_training_curves_jena.png
  06a_phase_kappa_synthetic.png
  06b_phase_kappa_occupancy.png
  06c_phase_kappa_power.png
  06d_phase_kappa_jena.png
  metrics_summary.csv
```
