"""Adversarial procedural guards using fake evidence, never reserved targets."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src import locality_protocol as p


CANONICAL_CONFIG = p.read_json(p.CONFIG_PATH)
# The file-backed adversarial fixture declares a small complete matrix. Exact
# real-program counts/order are tested separately without thousands of files.
CONFIG = copy.deepcopy(CANONICAL_CONFIG)
CONFIG["tasks"].update(families=["smooth","narrow"],periodic_families=["smooth","narrow"],
                        noise_levels=[.05],label_budgets=[128,512],
                        replicates={"pilot":1,"development":1,"confirmation":1},
                        initializations={"pilot":1,"development":1,"confirmation":1})
RATES = {model["id"]: .01 for model in CONFIG["models"] if model["kind"] != "ridge"}


@pytest.fixture(scope="module")
def evidence_tree(tmp_path_factory):
    """Build full canonical matrices with inert small files and declared scores."""
    root = tmp_path_factory.mktemp("locality_procedural_fixture")
    for name in p.SOURCE_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Nonexecutable source-binding fixture.\n")
    (root / "tests").mkdir()
    (root / "tests/test_fixture.py").write_text("# Inert full-suite provenance fixture\n")
    sources = p.source_manifest(root)

    def common(kind):
        return {"kind": kind, "complete": True, "config_hash": p.object_hash(CONFIG),
                "source_hashes": sources, "evidence_hashes": {}}

    def bind(record, path):
        record["evidence_hashes"][path.relative_to(root).as_posix()] = p.file_hash(path)

    def completion(phase,manifest):
        receipt=root/phase/"completion.json"
        p.write_json(receipt,{"manifest_hash":p.file_hash(root/phase/"manifest.json"),
                              "ended_unix":manifest["ended_unix"]+.1,"elapsed_seconds":10.1,
                              "deadline_respected":True},exclusive=False)

    model_errors = dict(fixed16=1.2, vp_centers=.8, vp_widths=.9, vp_both=1.1,
                        joint_both=1.3, fixed32=.85, fixed48=.75,
                        fourier16=1.4, fourier32=.95, fourier48=.96)
    manifests, summaries = {}, {}
    for phase in ("pilot", "development"):
        directory = root / phase
        cells = p.planned_runs(CONFIG, phase, None if phase == "pilot" else RATES)
        started = 10. if phase == "pilot" else 30.
        manifest = common("locality_run_v2")
        manifest.update(phase=phase, status="completed", deadline_respected=True,
                        planned_count=len(cells), completed_count=len(cells), learning_rates=None if phase == "pilot" else RATES,
                        started_unix=started, ended_unix=started+10, elapsed_seconds=10.,
                        deadline_unix=started+CONFIG["phases"][f"{phase}_ceiling_seconds"],
                        phase_ceiling_seconds=CONFIG["phases"][f"{phase}_ceiling_seconds"], rows=[])
        summaries[phase] = []
        for cell in cells:
            fitdir = directory / "fits" / cell["id"]
            fitdir.mkdir(parents=True)
            summary_path, artifact_path = fitdir / "summary.json", fitdir / "artifacts.pt"
            fit_seconds = .01 if cell["kind"] == "ridge" else 2.01
            validation = (0. if cell["role"] == "timing" else 1. + (cell["learning_rate"] or 0.))
            summary = {"model_id": cell["model_id"], "status": "completed", "best_validation_mse": validation,
                       "primary_error": model_errors[cell["model_id"]], "fit_seconds": fit_seconds,
                       "overshoot_seconds": 0. if cell["kind"] == "ridge" else .01}
            p.write_json(summary_path, summary)
            artifact_path.write_text("Inert bytes; no tensor contents, task generation or model execution.\n")
            task_path = directory / "tasks" / f"{cell['family']}_{cell['replicate']:03d}.json"
            if not task_path.exists():
                p.write_json(task_path, {"scope": "inert task-identity fixture only"})
            for path in (summary_path, artifact_path, task_path):
                bind(manifest, path)
            manifest["rows"].append({"cell": cell, "status": "completed",
                "summary_path": summary_path.relative_to(root).as_posix(), "summary_hash": p.file_hash(summary_path),
                "artifact_path": artifact_path.relative_to(root).as_posix(), "artifact_hash": p.file_hash(artifact_path),
                "task_path": task_path.relative_to(root).as_posix(), "fit_call_seconds": fit_seconds+.005,
                "wall_seconds": fit_seconds+.02})
            summaries[phase].append(summary)
        manifests[phase] = manifest
        p.write_json(directory / "manifest.json", manifest)
        completion(phase,manifest)

    pilot = manifests["pilot"]
    selection = common("locality_pilot_selection_v2")
    selection.update(input_manifest="pilot/manifest.json", learning_rates=RATES, scores={})
    for model_id in RATES:
        count=len(CONFIG["tasks"]["families"])*len(CONFIG["tasks"]["noise_levels"])
        selection["scores"][model_id] = [
            {"learning_rate": lr, "mean_validation_mse": float(np.mean([1.+lr]*count)), "count": count}
            for lr in CONFIG["training"]["pilot_learning_rates"]]
    selection["evidence_hashes"].update(pilot["evidence_hashes"])
    bind(selection, root / "pilot/manifest.json")
    bind(selection, root / "pilot/completion.json")
    p.write_json(root / "selection.json", selection)

    feasible = common("locality_feasibility_v2")
    overhead = float(np.quantile([row["wall_seconds"]-summary["fit_seconds"]
                                  for row,summary in zip(pilot["rows"],summaries["pilot"])], .9))
    fixed_time = float(np.quantile([row["fit_call_seconds"] for row in pilot["rows"]
                                    if row["cell"]["role"] == "timing" and row["cell"]["kind"] == "ridge"], .9))
    feasible.update(all_passed=True,input_manifest="pilot/manifest.json",pilot_selection="selection.json",
                    measured={"fixed_maxshape_call_p90":fixed_time,"nonfit_overhead_p90":overhead,"max_overshoot":.01},estimates={})
    for phase in ("development", "confirmation"):
        rows = p.planned_runs(CONFIG, phase, RATES)
        adaptive = sum(row["kind"] != "ridge" for row in rows)
        projected = adaptive * 2.01 + (len(rows)-adaptive) * fixed_time + len(rows)*overhead + 60
        feasible["estimates"][phase] = {"records":len(rows),"adaptive":adaptive,"fixed":len(rows)-adaptive,
            "projected_seconds":projected,"ceiling_seconds":CONFIG["phases"][f"{phase}_ceiling_seconds"],"passed":True}
    feasible["evidence_hashes"].update(selection["evidence_hashes"])
    bind(feasible,root/"selection.json")
    p.write_json(root/"feasibility.json",feasible)

    development = manifests["development"]
    development["prerequisite_hashes"] = {name:p.file_hash(root/name) for name in ("selection.json","feasibility.json")}
    p.write_json(root/"development/manifest.json",development,exclusive=False)
    completion("development",development)
    analysis = common("locality_analysis_v2")
    analysis.update(phase="development",input_manifest="development/manifest.json",learning_rates=RATES,
                    choices={"vp_model":"vp_centers","capacity_comparator":"fixed32"},audit={"all_passed":True},research_gate_passed=False)
    analysis["evidence_hashes"].update(development["evidence_hashes"])
    bind(analysis,root/"development/manifest.json")
    bind(analysis,root/"development/completion.json")
    p.write_json(root/"analysis.json",analysis)

    audit = common("locality_independent_audit_v2")
    audit.update(all_passed=True,checks={"numerical":True,"protocol":True},reports={})
    for name in ("numerical","protocol"):
        review_path = root / f"{name}_review.json"
        p.write_json(review_path,{"source_hashes":sources,"scope":"Disposable procedural proof fixture", "findings":[],"all_passed":True})
        audit["reports"][name]=review_path.relative_to(root).as_posix()
        bind(audit,review_path)
    p.write_json(root/"audit.json",audit)
    validation = common("locality_validation_v2")
    log = root/"full_suite.log"
    log.write_text("1234 passed in 12.34s\n")
    validation.update(all_passed=True,test_source_hashes={"tests/test_fixture.py":p.file_hash(root/"tests/test_fixture.py")},
        full_suite={"command":["py","-3.11","-m","pytest","tests/","-q"],"exit_code":0,"passed":1234,"failed":0,"errors":0,"log_path":"full_suite.log"})
    bind(validation,log)
    bind(validation,root/"tests/test_fixture.py")
    p.write_json(root/"validation.json",validation)
    inputs = {"pilot_selection":root/"selection.json","development_analysis":root/"analysis.json", "feasibility":root/"feasibility.json",
              "independent_audit":root/"audit.json","validation":root/"validation.json"}
    return root, inputs


@pytest.fixture
def tree(evidence_tree,monkeypatch):
    """Restore changed fixture bytes between tests without touching real sources."""
    root,inputs=evidence_tree
    # Test mutations are confined to these metadata files; fit evidence remains
    # immutable. Avoid rereading thousands of unchanged files between probes.
    mutable=[*inputs.values(),root/"pilot/manifest.json",root/"pilot/completion.json",root/"src/locality_models.py"]
    snapshot={path:path.read_bytes() for path in mutable}
    monkeypatch.setattr(p,"environment",lambda:{"scope":"fixture; no hardware query"})
    yield root,inputs
    for path in (root/"lock.json",root/"lock.json.sha256"):
        if path.exists():
            path.unlink()
    for path,data in snapshot.items():
        if path.read_bytes()!=data:
            path.write_bytes(data)


def mutate(path,change):
    value=p.read_json(path)
    change(value)
    p.write_json(path,value,exclusive=False)


def rehash_lock(path):
    path.with_suffix(path.suffix+".sha256").write_text(p.file_hash(path))


def test_complete_evidence_and_failed_research_gate_can_create_verified_lock(tree):
    root,inputs=tree
    lock=p.create_confirmation_lock(root/"lock.json",CONFIG,inputs,root)
    assert lock["lock_sha256"]==p.file_hash(root/"lock.json")
    assert "lock_sha256" not in p.read_json(root/"lock.json")
    assert lock["learning_rates"]==RATES
    assert lock["choices"]=={"vp_model":"vp_centers","capacity_comparator":"fixed32"}
    assert p.verify_confirmation_lock(root/"lock.json",CONFIG,root)==lock
    assert p.read_json(inputs["development_analysis"])["research_gate_passed"] is False


@pytest.mark.parametrize("phase,counts",[("pilot",(224,128,96)),("development",(1440,576,864)),("confirmation",(3840,1536,2304))])
def test_canonical_matrices_are_exact_and_do_not_draw_global_rng(phase,counts):
    before=np.random.get_state()
    rows=p.planned_runs(CANONICAL_CONFIG,phase,RATES)
    assert len(rows)==counts[0]
    assert sum(row["kind"]!="ridge" for row in rows)==counts[1]
    assert sum(row["kind"]=="ridge" for row in rows)==counts[2]
    assert len({row["id"] for row in rows})==len(rows)
    assert rows==p.planned_runs(CANONICAL_CONFIG,phase,RATES)
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


@pytest.mark.parametrize("rates",[None,{},dict(RATES,extra=.01),dict(RATES,vp_both=-1),dict(RATES,vp_both=.02),
                                  dict(RATES,vp_both=True),dict(RATES,vp_both=float("nan")),dict(RATES,vp_both=".01")])
def test_incomplete_invalid_or_unpiloted_rate_maps_rejected(rates):
    with pytest.raises(ValueError):
        p.planned_runs(CONFIG,"development",rates)


@pytest.mark.parametrize("choices",[{}, {"vp_model":"joint_both","capacity_comparator":"fixed48"},
    {"vp_model":"vp_centers","capacity_comparator":"fixed48"},{"vp_model":"vp_both","capacity_comparator":"vp_both"},
    {"vp_model":[],"capacity_comparator":"fixed32"}])
def test_invalid_choices_rejected(choices):
    with pytest.raises(ValueError):
        p.validate_choices(choices,CONFIG)


@pytest.mark.parametrize("change",[
    lambda x:x["learning_rates"].update(vp_both=.1),
    lambda x:x.update(choices={"vp_model":"vp_both","capacity_comparator":"fixed48"}),
    lambda x:x["evidence_hashes"].pop("selection.json"),
    lambda x:x["evidence_hashes"].pop(next(name for name in x["evidence_hashes"] if name.endswith("artifacts.pt"))),
    lambda x:x.update(lock_sha256="invented"),
])
def test_rehashed_lock_still_rejects_semantic_or_closure_changes(tree,change):
    root,inputs=tree
    path=root/"lock.json"
    p.create_confirmation_lock(path,CONFIG,inputs,root)
    mutate(path,change)
    rehash_lock(path)
    with pytest.raises(ValueError):
        p.verify_confirmation_lock(path,CONFIG,root)


@pytest.mark.parametrize("change",[
    lambda x:x.update(kind="unrelated_complete_record"),
    lambda x:x["learning_rates"].update(vp_both=.1),
    lambda x:x["scores"]["vp_both"].pop(),
    lambda x:x["scores"]["vp_both"][0].update(mean_validation_mse=0),
    lambda x:x["evidence_hashes"].pop("pilot/manifest.json"),
])
def test_pilot_proof_requires_complete_scores_and_exact_selection(tree,change):
    root,inputs=tree
    mutate(inputs["pilot_selection"],change)
    with pytest.raises(ValueError):
        p.verify_pilot_selection(inputs["pilot_selection"],CONFIG,root)


@pytest.mark.parametrize("change",[
    lambda x:x["estimates"].pop("confirmation"),
    lambda x:x["estimates"]["confirmation"].update(projected_seconds=1),
    lambda x:x["estimates"]["confirmation"].update(ceiling_seconds=100000),
    lambda x:x["measured"].update(nonfit_overhead_p90=0),
    lambda x:x.update(input_manifest="development/manifest.json"),
])
def test_feasibility_requires_both_exact_same_pilot_projections(tree,change):
    root,inputs=tree
    mutate(inputs["feasibility"],change)
    with pytest.raises(ValueError):
        p.verify_feasibility(inputs["feasibility"],CONFIG,root)


@pytest.mark.parametrize("change",[
    lambda x:x.update(phase="confirmation"),
    lambda x:x.update(audit={"all_passed":False}),
    lambda x:x.update(learning_rates=dict(RATES,vp_both=.1)),
    lambda x:x.update(choices={"vp_model":"vp_widths","capacity_comparator":"fixed32"}),
])
def test_development_requires_audited_actual_selected_choices_and_rates(tree,change):
    root,inputs=tree
    mutate(inputs["development_analysis"],change)
    with pytest.raises(ValueError):
        p.create_confirmation_lock(root/"lock.json",CONFIG,inputs,root)
    assert not (root/"lock.json").exists()


@pytest.mark.parametrize("change",[
    lambda x:x.update(kind="unrelated_complete_record"),
    lambda x:x.update(test_source_hashes={}),
    lambda x:x["full_suite"].update(command=["py","-3.11","-m","pytest","tests/test_fixture.py","-q"]),
    lambda x:x["full_suite"].update(passed=1),
    lambda x:x["full_suite"].update(exit_code=False),
    lambda x:x["evidence_hashes"].pop("full_suite.log"),
])
def test_validation_requires_current_all_test_sources_and_matching_full_log(tree,change):
    root,inputs=tree
    mutate(inputs["validation"],change)
    with pytest.raises(ValueError):
        p.create_confirmation_lock(root/"lock.json",CONFIG,inputs,root)


@pytest.mark.parametrize("change",[
    lambda x:x.update(checks={"numerical":True}),
    lambda x:x.update(reports={"numerical":"numerical_review.json","protocol":"numerical_review.json"}),
    lambda x:x["evidence_hashes"].pop("protocol_review.json"),
])
def test_independent_audit_needs_distinct_bound_source_specific_reviews(tree,change):
    root,inputs=tree
    mutate(inputs["independent_audit"],change)
    with pytest.raises(ValueError):
        p.create_confirmation_lock(root/"lock.json",CONFIG,inputs,root)


@pytest.mark.parametrize("change",[
    lambda x:x["rows"].pop(),
    lambda x:x["rows"].append(x["rows"][0]),
    lambda x:x["rows"][0].update(status="failed"),
    lambda x:x.update(ended_unix=x["deadline_unix"]+1),
    lambda x:x.update(deadline_unix=x["deadline_unix"]+100),
    lambda x:x["rows"][0].update(summary_hash="0"*64),
    lambda x:x["evidence_hashes"].pop(x["rows"][0]["summary_path"]),
])
def test_run_matrix_rejects_missing_duplicate_failed_unbound_or_late_records(tree,change):
    root,_=tree
    path=root/"pilot/manifest.json"
    mutate(path,change)
    with pytest.raises(ValueError):
        p.verify_run_manifest(path,CONFIG,root,"pilot")


def test_missing_or_changed_lock_and_sources_are_rejected_without_overwrite(tree):
    root,inputs=tree
    with pytest.raises(ValueError):
        p.verify_confirmation_lock(None,CONFIG,root)
    path=root/"lock.json"
    p.create_confirmation_lock(path,CONFIG,inputs,root)
    original=path.read_bytes()
    with pytest.raises(FileExistsError):
        p.create_confirmation_lock(path,CONFIG,inputs,root)
    assert path.read_bytes()==original
    (root/"src/locality_models.py").write_text("changed source\n")
    with pytest.raises(ValueError,match="Source"):
        p.verify_confirmation_lock(path,CONFIG,root)


def test_evidence_path_traversal_is_rejected(tree):
    root,_=tree
    with pytest.raises(ValueError,match="escapes"):
        p.checked_path("../outside.txt",root)


@pytest.mark.parametrize("change",[
    lambda x:x.update(manifest_hash="0"*64),
    lambda x:x.update(ended_unix=9999,elapsed_seconds=9989),
    lambda x:x.update(deadline_respected=False),
])
def test_completion_receipt_covers_full_manifest_flush_and_exact_hash(tree,change):
    root,_=tree
    mutate(root/"pilot/completion.json",change)
    with pytest.raises(ValueError,match="Final manifest"):
        p.verify_run_manifest(root/"pilot/manifest.json",CONFIG,root,"pilot")


def test_hash_memoization_is_within_call_only_and_invalidates_changed_evidence(tree,monkeypatch):
    root,_=tree
    calls=[]
    original=p._raw_file_hash
    monkeypatch.setattr(p,"_raw_file_hash",lambda path:(calls.append(path.resolve()),original(path))[1])
    p.verify_pilot_selection(root/"selection.json",CONFIG,root)
    assert len(calls)==len(set(calls))
    first=len(calls)
    p.verify_pilot_selection(root/"selection.json",CONFIG,root)
    assert len(calls)==2*first


def test_within_call_changed_file_is_rehashed(tree,monkeypatch):
    root,_=tree
    path=root/"src/locality_models.py"
    calls=[]
    original=p._raw_file_hash
    monkeypatch.setattr(p,"_raw_file_hash",lambda name:(calls.append(name),original(name))[1])
    @p._memoized_evidence
    def probe():
        first=p.file_hash(path)
        path.write_text("Changed while the enclosing verification scope remains active.\n")
        second=p.file_hash(path)
        return first,second
    first,second=probe()
    assert first!=second and len(calls)==2
