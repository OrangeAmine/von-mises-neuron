"""Execute the approved VM protocol with recorded cells and locked confirmation."""

from __future__ import annotations

import argparse
import _thread
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.methodology_protocol import (
    CONFIG_PATH, environment, file_hash, object_hash, planned_runs, read_json,
    snapshot_sources, source_manifest, verify_confirmation_lock,
    write_confirmation_lock, write_json,
)


@contextmanager
def invocation_watchdog(output: Path, deadline_unix: float):
    """Interrupt at the immutable ceiling; allow only bounded artifact cleanup."""
    finished = threading.Event()
    def watch() -> None:
        if not finished.wait(max(0,deadline_unix-time.time())):
            print("Declared invocation ceiling reached; interrupting and retaining incomplete evidence",flush=True)
            _thread.interrupt_main()
            if not finished.wait(2):
                # A stuck native call may not deliver Python's interrupt. Its
                # already registered running/pending cells remain on disk.
                write_json(output/"hard_timeout.json",{"status":"hard_timeout", "deadline_unix":deadline_unix})
                os._exit(124)
    watcher = threading.Thread(target=watch,daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join(timeout=1)


def _choices(path: Path | None, config: dict[str, Any]) -> dict[str, Any] | None:
    """Require pilot choices to remain bound to the code and measured evidence."""
    if path is None:
        return None
    selection = read_json(path)
    if selection["config_hash"] != object_hash(config) or selection["source_hashes"] != source_manifest():
        raise ValueError("Pilot selection source or configuration changed")
    for name,digest in selection["pilot_manifest_hashes"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("Pilot evidence changed")
    return selection["pilot_hyperparameters"]


def _pilot_training_sources_match(recorded: dict[str, str], current: dict[str, str]) -> bool:
    """Permit documented pre-lock audit/orchestration repairs, never fit changes.

    Original pilot snapshots remain immutable. Every model, trainer, generator,
    canonical plan/config and numerical source must still match byte for byte.
    Full current source equality remains mandatory after confirmation locking.
    """
    observers = {"scripts/run_vm_methodology.py", "scripts/analyze_vm_methodology.py"}
    return (set(recorded) == set(current)
            and all(recorded[name] == digest for name,digest in current.items() if name not in observers))


def _save_task(directory: Path, task: Any) -> dict[str, str]:
    """Archive exact shared arrays and the declared function parameters."""
    directory.mkdir(parents=True,exist_ok=False)
    arrays = task.as_arrays()
    np.savez_compressed(directory/"data.npz", **arrays)
    write_json(directory/"manifest.json",task.to_manifest())
    return {name: file_hash(directory/name) for name in ("data.npz","manifest.json")}


def _tensor_artifact(value: Any) -> Any:
    """Keep artifact checkpoints loadable with PyTorch weights_only=True."""
    if isinstance(value,np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value,dict):
        return {key:_tensor_artifact(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [_tensor_artifact(item) for item in value]
    if isinstance(value,np.generic):
        return value.item()
    return value


def progress_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Avoid duplicating immutable detailed summaries in every progress update.

    Running manifests keep the complete registered matrix, statuses, flags,
    artifact hashes and summary references. Final manifests retain the original
    full row schema. The supplied in-memory manifest is never mutated.
    """
    payload = dict(manifest)
    lightweight = manifest["status"] == "running"
    payload["progress_representation"] = (
        "lightweight_summary_references_v1" if lightweight else "full_summaries_v1"
    )
    if lightweight:
        rows = []
        for original in manifest["runs"]:
            row = {key: value for key, value in original.items() if key != "summary"}
            if "summary" in original:
                reference = f"fits/{original['run_id']}/summary.json"
                if (reference in original.get("artifact_hashes", {})
                        and original["summary"]["status"] == original["status"]):
                    row["summary_ref"] = reference
                else:
                    # An exception between artifact writes and row registration
                    # must retain any partial in-memory evidence inline.
                    row["summary"] = original["summary"]
            rows.append(row)
        payload["runs"] = rows
    return payload


def hydrate_progress_manifest(manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    """Restore hash-verified detailed summaries before resuming progress.

    Historical full manifests remain supported. Missing, altered or misplaced
    summary references are rejected before any task is generated or fit runs.
    """
    representation = manifest.get("progress_representation", "full_summaries_v1")
    if representation not in {"lightweight_summary_references_v1", "full_summaries_v1"}:
        raise ValueError("Unknown progress manifest representation")
    hydrated = dict(manifest)
    rows = []
    for original in manifest["runs"]:
        row = dict(original)
        reference = row.pop("summary_ref", None)
        if reference is not None:
            expected = f"fits/{row['run_id']}/summary.json"
            if reference != expected or reference not in row.get("artifact_hashes", {}):
                raise ValueError("Invalid or unhashed progress summary reference")
            path = (output / reference).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError("Progress summary reference outside invocation directory")
            if not path.is_file() or file_hash(path) != row["artifact_hashes"][reference]:
                raise ValueError("Changed or missing progress summary artifact")
            summary = read_json(path)
            if "summary" in row and row["summary"] != summary:
                raise ValueError("Progress summary differs from its immutable artifact")
            if summary["status"] != row["status"]:
                raise ValueError("Progress status differs from immutable summary")
            row["summary"] = summary
        elif representation == "lightweight_summary_references_v1":
            expected = f"fits/{row['run_id']}/summary.json"
            if expected in row.get("artifact_hashes", {}) and "summary" not in row:
                raise ValueError("Lightweight progress row omitted its summary reference")
        rows.append(row)
    hydrated["runs"] = rows
    hydrated["progress_representation"] = "full_summaries_v1"
    return hydrated


def _fit(row: dict[str, Any], task: Any, config: dict[str, Any], deadline: float,
         device: torch.device) -> dict[str, Any]:
    """Dispatch one prespecified fit; no outcome-dependent hyperparameters."""
    from src.methodology_tasks import initialization_bundle, batch_seed, cyclic_batch_indices
    from src.methodology_training import make_a_model, make_b_model, train_a, train_b, fit_ridge, closed_form_a
    if row["stage"] == "A":
        if row["kind"] == "reference":
            return closed_form_a(task,penalty=row["penalty"])
        parameters = task.function.parameters
        model = make_a_model(row["model"], parameters["amplitude_before"], parameters["phase_before"],device=device)
        return train_a(model,task,optimizer_name=row["optimizer"],learning_rate=row["learning_rate"],
            penalty=row["penalty"], updates=config["A"]["pilot"]["updates"] if row["phase"] == "pilot" else config["A"]["updates"],
            record_every=config["A"]["record_every"],device=device,deadline=deadline)
    task = task.train_prefix(row["n_train"])
    if row["kind"] == "ridge":
        return fit_ridge(row["model"],task,lambdas=config["B"]["ridge"]["lambda_grid"],deadline=deadline)
    initialization = initialization_bundle(config,task,row["initialization_index"])
    model = make_b_model(row["model"],initialization,device=device)
    options = config["B"]["optimizer"]
    updates = config["B"]["pilot"]["max_updates"] if row["phase"] == "pilot" else options["max_updates"]
    batches = cyclic_batch_indices(row["n_train"],options["batch_size"],updates,
        batch_seed(config,task,row["initialization_index"]))
    bundle = train_b(model,task,learning_rate=row["learning_rate"],batch_indices=batches,
        batch_seed=batch_seed(config,task,row["initialization_index"]),
        max_updates=updates,batch_size=options["batch_size"],validation_every=options["validation_every"],
        patience_evaluations=options["patience_evaluations"],min_delta=options["min_delta"],
        readout_penalty=options["readout_penalty"],device=device,deadline=deadline)
    bundle["summary"]["initialization_metadata"] = initialization["metadata"]
    bundle["artifacts"]["initialization_bundle"] = _tensor_artifact(initialization)
    return bundle


def execute_stage(config: dict[str, Any], stage: str, phase: str, output: Path,
                  *, selection: Path | None = None, lock: Path | None = None,
                  feasibility: Path | None = None, resume: bool = False) -> dict[str, Any]:
    """Run one fixed matrix serially, retaining failed and interrupted records."""
    lock_data = verify_confirmation_lock(lock,config) if phase == "confirmation" else None
    choices = lock_data["pilot_hyperparameters"] if lock_data else _choices(selection,config)
    if phase != "pilot" and choices is None:
        raise ValueError("Development requires prescribed pilot selection")
    if phase == "development":
        if feasibility is None:
            raise ValueError("Development requires measured pilot feasibility")
        assessment = read_json(feasibility)
        if not assessment["feasible"] or assessment["config_hash"] != object_hash(config):
            raise ValueError("Full matrix is not feasible under the unchanged contract")
        for name,digest in assessment["pilot_manifest_hashes"].items():
            if file_hash(Path(name)) != digest:
                raise ValueError("Feasibility pilot evidence changed")
    rows = planned_runs(config,stage,phase,choices)
    minutes = config["execution"]["timeouts_minutes"][f"{stage}_{phase}"]
    path = output/"manifest.json"
    if resume:
        manifest = read_json(path)
        if manifest["source_hashes"] != source_manifest() or manifest["config_hash"] != object_hash(config):
            raise ValueError("Cannot resume under changed source or configuration")
        manifest = hydrate_progress_manifest(manifest, output)
        expected = [{k:v for k,v in row.items() if k != "status"} for row in rows]
        actual = [{k:row[k] for k in expected[i]} for i,row in enumerate(manifest["runs"])]
        if actual != expected:
            raise ValueError("Cannot change a registered matrix on resume")
        for row in manifest["runs"]:
            if row["status"] == "running":
                row.update(status="interrupted",error="Prior process ended during this fit; retained, not retried")
        for row in manifest["runs"]:
            if "artifact_hashes" in row:
                for name,digest in row["artifact_hashes"].items():
                    if file_hash(output/name) != digest:
                        raise ValueError("Changed prior fit artifact")
    else:
        output.mkdir(parents=True,exist_ok=False)
        manifest = {"protocol_id": config["protocol_id"], "stage":stage,"phase":phase,
            "config_hash":object_hash(config),"config":config,"source_hashes":snapshot_sources(output),
            "environment":environment(),"pilot_hyperparameters":choices,
            "confirmation_lock":str(lock) if lock else None,
            "confirmation_lock_sha256":lock_data["lock_sha256"] if lock_data else None,
            "status":"prepared","progress_representation":"full_summaries_v1",
            "time_seconds":0.,"ceiling_minutes":minutes,"runs":rows,
            "deadline_unix":time.time()+minutes*60,
            "resume_budget":"Original absolute invocation deadline remains fixed, including downtime",
            "tasks":{}, "raw_data":"none; declared synthetic functions", "units":"angles radians; A unstandardized signal; B generator-standardized signal"}
        write_json(path,manifest)
    # Resumption shares the original invocation ceiling, never a renewed budget.
    started = time.perf_counter()
    previous_elapsed = manifest["time_seconds"]
    deadline = started + max(0,min(minutes*60-previous_elapsed,manifest["deadline_unix"]-time.time()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.methodology_tasks import generate_task
    cache: dict[str, Any] = {}
    manifest["status"] = "running"
    def persist() -> None:
        manifest["time_seconds"] = previous_elapsed + time.perf_counter()-started
        payload = progress_manifest_payload(manifest)
        manifest["progress_representation"] = payload["progress_representation"]
        write_json(path,payload,exclusive=False)
    try:
        persist()
        for row in manifest["runs"]:
            if row["status"] != "pending":
                continue
            if time.perf_counter() >= deadline:
                row.update(status="not_run_timeout",error="Declared invocation ceiling exhausted")
                persist()
                continue
            row["status"] = "running"
            persist()
            begin = time.perf_counter()
            fit_dir = output/"fits"/row["run_id"]
            fit_dir.mkdir(parents=True,exist_ok=False)
            try:
                key = f"{row['family']}_{row['replicate']}"
                if key not in cache:
                    task = generate_task(config,stage,phase,row["family"],row["replicate"],confirmation_lock=lock)
                    cache[key] = task
                    task_dir = output/"tasks"/key
                    if key not in manifest["tasks"]:
                        manifest["tasks"][key] = _save_task(task_dir,task)
                    else:
                        for name,digest in manifest["tasks"][key].items():
                            if file_hash(task_dir/name) != digest:
                                raise ValueError("Changed task artifact")
                bundle = _fit(row,cache[key],config,deadline,device)
                summary = bundle["summary"]
                torch.save(bundle["artifacts"],fit_dir/"artifacts.pt")
                write_json(fit_dir/"summary.json",summary)
                row.update(status=summary["status"],summary=summary)
                row["artifact_hashes"] = {str((fit_dir/name).relative_to(output)).replace("\\","/"):file_hash(fit_dir/name) for name in ("artifacts.pt","summary.json")}
                if row["status"] == "interrupted":
                    raise KeyboardInterrupt("Fit interruption retained with partial evidence")
            except Exception as exc:
                row.update(status="failed",error=f"{type(exc).__name__}: {exc}")
                write_json(fit_dir/"failure.json",{"error":row["error"]})
            except BaseException as exc:
                row.update(status="interrupted",error=f"{type(exc).__name__}: {exc}")
                manifest["status"] = "interrupted"
                persist()
                raise
            finally:
                row["wall_seconds"] = time.perf_counter()-begin
                persist()
            print(f"{row['run_id']} {row['family']} {row['model']} {row['status']} {row['wall_seconds']:.3f}s",flush=True)
        manifest["status"] = "completed" if all(row["status"] == "completed" for row in manifest["runs"]) else "incomplete"
        persist()
    except BaseException:
        # Graceful interruption between fits still gets the full final schema.
        # An abrupt process kill leaves a hash-verifiable lightweight checkpoint.
        if manifest["status"] != "interrupted":
            manifest["status"] = "interrupted"
            persist()
        raise
    return manifest


def select_pilots(config: dict[str, Any], directories: list[Path]) -> dict[str, Any]:
    """Select eligible complete rates using the prescribed noisy validation scores."""
    result: dict[str, Any] = {"pilot_hyperparameters":{"A":{},"B":{}},"candidates":{},"pilot_manifest_hashes":{},
                             "config_hash":object_hash(config),"source_hashes":source_manifest()}
    for directory in directories:
        manifest = read_json(directory/"manifest.json")
        if manifest["phase"] != "pilot" or manifest["config_hash"] != object_hash(config):
            raise ValueError("Not a matching pilot manifest")
        if not _pilot_training_sources_match(manifest["source_hashes"],result["source_hashes"]):
            raise ValueError("Pilot training sources changed before selection")
        result.setdefault("prelock_observer_source_changes",{})[manifest["stage"]] = {
            name:{"pilot_sha256":digest,"current_sha256":result["source_hashes"][name]}
            for name,digest in manifest["source_hashes"].items() if digest != result["source_hashes"][name]}
        stage = manifest["stage"]
        rows = [row for row in manifest["runs"] if row["kind"] == "gradient"]
        groups = list(config["A"]["optimizers"]) if stage == "A" else [m["id"] for m in config["B"]["models"] if m["kind"] == "gradient"]
        result["candidates"][stage] = {}
        for group in groups:
            rates = config["A"]["optimizers"][group]["pilot_learning_rates"] if stage == "A" else config["B"]["optimizer"]["pilot_learning_rates"]
            candidates = []
            for rate in rates:
                cells = [row for row in rows if row["learning_rate"] == rate and row["optimizer" if stage == "A" else "model"] == group]
                expected = 8 if stage == "A" else 4
                eligible = len(cells) == expected and all(row["status"] == "completed" for row in cells)
                scores = []
                if eligible:
                    for row in cells:
                        summary = row["summary"]
                        scores.append(summary["validation_curve_area_normalized"] if stage == "A" else summary["best_validation_mse"])
                    eligible = bool(np.isfinite(scores).all())
                candidates.append({"learning_rate":rate,"eligible":eligible,"mean_score":float(np.mean(scores)) if eligible else None})
            complete = [item for item in candidates if item["eligible"]]
            if not complete:
                raise ValueError(f"No complete eligible rate for {stage}/{group}; design revision required")
            best = min(item["mean_score"] for item in complete)
            selected = min(item["learning_rate"] for item in complete if item["mean_score"] <= best+1e-12)
            result["pilot_hyperparameters"][stage][group] = selected
            result["candidates"][stage][group] = candidates
        result["pilot_manifest_hashes"][str((directory/"manifest.json").resolve())] = file_hash(directory/"manifest.json")
    if set(result["candidates"]) != {"A","B"}:
        raise ValueError("Both prescribed pilots are required")
    return result


def pilot_feasibility(config: dict[str, Any], directories: list[Path]) -> dict[str, Any]:
    """Extrapolate measured fit times at full update counts without reducing budgets."""
    rows = []
    hashes = {}
    selected_choices = select_pilots(config,directories)["pilot_hyperparameters"]
    for directory in directories:
        manifest = read_json(directory/"manifest.json")
        if manifest["phase"] != "pilot" or manifest["config_hash"] != object_hash(config) or not _pilot_training_sources_match(manifest["source_hashes"],source_manifest()):
            raise ValueError("Feasibility requires matching current pilot evidence")
        hashes[str((directory/"manifest.json").resolve())] = file_hash(directory/"manifest.json")
        stage = manifest["stage"]
        times = [row["wall_seconds"] for row in manifest["runs"] if row["kind"] == "gradient" and row["status"] == "completed"]
        expected = config[stage]["pilot"]["gradient_fits"]
        if len(times) != expected:
            raise ValueError("Incomplete pilot cannot establish full-matrix feasibility")
        ratio = config["A"]["updates"]/config["A"]["pilot"]["updates"] if stage == "A" else config["B"]["optimizer"]["max_updates"]/config["B"]["pilot"]["max_updates"]
        for phase in ("development","confirmation"):
            count = config[stage][phase]["gradient_fits"]
            median = float(np.median(times))*ratio*count
            p90 = float(np.percentile(times,90))*ratio*count
            other = [row["wall_seconds"] for row in manifest["runs"] if row["kind"] != "gradient"]
            other_count = config[stage][phase].get("ridge_configurations", config[stage][phase]["task_pairs"]*8)
            # B closed-form cost may grow with sample size; conservatively scale
            # every future solve by the largest n/pilot n ratio.
            other_ratio = max(config["B"]["train_observations"])/config["B"]["pilot"]["train_observations"] if stage == "B" else 1
            other_cost = float(np.percentile(other,90))*other_count*other_ratio if other else 0
            overhead = max(0,manifest["time_seconds"]-sum(row["wall_seconds"] for row in manifest["runs"]))
            median += other_cost+overhead
            p90 += other_cost+overhead
            pooled_median, pooled_p90 = median,p90
            strata = []
            planned = planned_runs(config,stage,phase,selected_choices)
            group_key = "optimizer" if stage == "A" else "model"
            median,p90 = other_cost+overhead,other_cost+overhead
            for group,rate in selected_choices[stage].items():
                values = [row["wall_seconds"] for row in manifest["runs"] if row["kind"] == "gradient"
                          and row[group_key] == group and row["learning_rate"] == rate]
                expected_count = 8 if stage == "A" else 4
                if len(values) != expected_count:
                    raise ValueError("Stratified timing must retain every selected-rate pilot cell")
                fits = sum(row["kind"] == "gradient" and row[group_key] == group for row in planned)
                group_median = float(np.median(values))*ratio*fits
                group_p90 = float(np.percentile(values,90))*ratio*fits
                median += group_median
                p90 += group_p90
                strata.append({"group":group,"validation_selected_learning_rate":rate,
                    "all_pilot_fit_seconds":values,"planned_fits":fits,
                    "projected_median_seconds":group_median,"projected_p90_seconds":group_p90})
            ceiling = 60*config["execution"]["timeouts_minutes"][f"{stage}_{phase}"]
            rows.append({"stage":stage,"phase":phase,"fits":count,"pilot_median_fit_seconds":float(np.median(times)),
                "pilot_p90_fit_seconds":float(np.percentile(times,90)),"update_ratio":ratio,
                "projected_closed_form_and_overhead_seconds":other_cost+overhead,
                "pooled_all_rate_median_seconds":pooled_median,"pooled_all_rate_p90_seconds":pooled_p90,
                "fixed_composition_strata":strata,
                "projected_median_seconds":median,"projected_p90_seconds":p90,"ceiling_seconds":ceiling,
                "fits_ceiling":p90 < ceiling})
    return {"method":"Fixed-composition model/optimizer strata at validation-selected rates; every pilot cell in each selected stratum retained, full-update extrapolation plus conservative closed-form and observed overhead",
            "caution":"Sparse timing samples; p90 is not a probabilistic runtime guarantee. Original pooled all-rate estimates remain visible. Invocation ceilings unchanged.",
            "config_hash":object_hash(config),"pilot_manifest_hashes":hashes,
            "projections":rows,"feasible":len(rows)==4 and all(row["fits_ceiling"] for row in rows)}


def main() -> int:
    """Expose separate run, pilot selection, feasibility and lock operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=CONFIG_PATH)
    sub = parser.add_subparsers(dest="command",required=True)
    run = sub.add_parser("run")
    run.add_argument("--stage",choices=["A","B"],required=True)
    run.add_argument("--phase",choices=["pilot","development","confirmation"],required=True)
    run.add_argument("--output-dir",type=Path,required=True)
    run.add_argument("--selection",type=Path)
    run.add_argument("--lock",type=Path)
    run.add_argument("--feasibility",type=Path)
    run.add_argument("--resume",action="store_true")
    for command in ("select-pilots","feasibility"):
        child = sub.add_parser(command)
        child.add_argument("--pilots",type=Path,nargs=2,required=True)
        child.add_argument("--output",type=Path,required=True)
    child = sub.add_parser("lock")
    child.add_argument("--selection",type=Path,required=True)
    child.add_argument("--development-analysis",type=Path,required=True)
    child.add_argument("--feasibility",type=Path,required=True)
    child.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    with threadpool_limits(limits=1):
        if args.command == "run":
            deadline = (read_json(args.output_dir/"manifest.json")["deadline_unix"] if args.resume else
                        time.time()+60*config["execution"]["timeouts_minutes"][f"{args.stage}_{args.phase}"])
            with invocation_watchdog(args.output_dir,deadline):
                result = execute_stage(config,args.stage,args.phase,args.output_dir,selection=args.selection,lock=args.lock,feasibility=args.feasibility,resume=args.resume)
            return 0 if result["status"] == "completed" else 1
        if args.command in ("select-pilots","feasibility"):
            result = select_pilots(config,args.pilots) if args.command == "select-pilots" else pilot_feasibility(config,args.pilots)
            write_json(args.output,result)
            print(result,flush=True)
            return 0 if result.get("feasible",True) else 1
        analysis = read_json(args.development_analysis)
        feasibility = read_json(args.feasibility)
        selection_data = read_json(args.selection)
        if (not analysis["complete"] or analysis["phase"] != "development"
                or {audit["stage"] for audit in analysis["audits"]} != {"A","B"}
                or not all(audit["complete"] for audit in analysis["audits"])
                or analysis["A"] is None or analysis["B"] is None
                or not feasibility["feasible"] or feasibility["config_hash"] != object_hash(config)
                or feasibility["pilot_manifest_hashes"] != selection_data["pilot_manifest_hashes"]):
            raise ValueError("Lock requires complete audited development and feasible full matrix")
        dependencies = [args.selection,args.development_analysis,args.feasibility]
        dependencies.extend(Path(name) for name in analysis["input_manifest_hashes"])
        dependencies.extend(Path(name) for name in read_json(args.selection)["pilot_manifest_hashes"])
        development_source_bridges = {}
        for name,digest in analysis["input_manifest_hashes"].items():
            if file_hash(Path(name)) != digest:
                raise ValueError("Development evidence changed after analysis")
            manifest = read_json(Path(name))
            current_sources = source_manifest()
            if not _pilot_training_sources_match(manifest["source_hashes"], current_sources) or manifest["phase"] != "development":
                raise ValueError("Development training source changed or wrong study phase")
            changed = {source:{"development_sha256":previous,"current_sha256":current_sources[source]}
                       for source,previous in manifest["source_hashes"].items() if previous != current_sources[source]}
            if changed:
                development_source_bridges[name] = {"manifest_sha256":digest, "observer_changes":changed}
        if development_source_bridges:
            bridge_path = args.output.with_name(args.output.name + ".prelock_source_bridge.json")
            write_json(bridge_path, {
                "scope":"Prelock observer/storage repairs; model, trainer, generator, canonical plan/config and numerical source bytes unchanged",
                "confirmation_rule":"All current source hashes are frozen in the confirmation lock; no observer exemption after locking",
                "development_sources":development_source_bridges,
            })
            dependencies.append(bridge_path)
        write_confirmation_lock(args.output,config,_choices(args.selection,config),analysis["B_primary_comparator"],dependencies)
        print(f"Confirmation locked: {args.output}",flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
