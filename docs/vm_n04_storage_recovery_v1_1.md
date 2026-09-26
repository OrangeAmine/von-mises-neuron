# VM-N04 storage-only recovery, before development or confirmation

The original `study_v1` pilot completed all 48 planned neural fits and saved
their checkpoints, 192 prediction arrays and completed per-fit journals. It
then failed serializing the final summary: irregular-attendance metadata held
NumPy int32 day-start indices. The original directory and original eight-file
source lock remain unchanged. Development access was blocked; no development
or confirmation trajectories had been generated.

The v1.1 controller leaves every original scientific source byte unchanged and
only supplies JSON conversion of NumPy numeric scalars. Its source and this
receipt are added to a successor lock before further trajectories. The
scientific v1 configuration, model, data, optimization, budgets, selection,
metrics and gates are unchanged. This is an explicit orchestration deviation
from uninterrupted v1 execution, not an unreported successful original pilot.

Recovery verifies the original artifact hashes, planned fit identities,
validation-selected checkpoint epochs, original saved predictions and metrics.
It reconstructs only already-consumed pilot metadata, checking saved model
inputs and targets exactly. It copies completed fit evidence without refitting,
retrying any fit or selecting using pilot test error. The common learning rate
comes solely from the original validation histories.

The failed summary lost its in-memory monotonic stage timer and reference-only
fit/inference timers. Those reference times stay null. Feasibility conservatively
uses elapsed wall time from the original access receipt to its last artifact,
plus a 60-second allowance and measured recovery time, followed by the unchanged
1.5 multiplier and unchanged later-stage caps. This is a reconstructed upper
estimate, not a recovered exact monotonic measurement. Saved per-neural-fit
training and inference times remain actual original measurements.

The separate recovery receipt binds the original lock, original artifact
manifest, original initial summary, selected rate and timing method. The new
pilot access receipt binds this recovery record. Development/confirmation use
the normal completed-stage, exact-matrix, source and artifact checks. If any
original evidence is missing or inconsistent, recovery fails instead of
creating missing fitted results. No reserved seeds are used by recovery.
