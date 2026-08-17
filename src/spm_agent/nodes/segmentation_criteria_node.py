from spm_agent.config import criteria_dir
from spm_agent.prompts.segmentation_criteria_prompts import (
    build_criteria_system_message, build_criteria_human_message)
from spm_agent.schemas.segmentation_criteria import validate_criteria

from spm_agent.tools.run_python import make_run_python, LocalBackend, extract_run_python_code
from spm_agent.utils.message_utils import flatten_text
from spm_agent.config import SANDBOX_PY, SEG_MODEL, SEG_MAX_TOKENS, SEG_MAX_SUPERSTEPS

from spm_agent.states.image_analysis_state import AnalysisState

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from pathlib import Path
import shutil
import numpy as np

MAX_VALIDATION_RETRIES = 2



def _make_workdir(importance_dir) -> Path:
    wd = importance_dir / "work"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    return wd


async def segmentation_criteria_node(state: AnalysisState) -> AnalysisState:
    tasks    = state.get("experiment_tasks", [])
    channels = state["file_channels"]           # type: ignore
    seg      = state["segmentation_results"]     # type: ignore
    grid_hw  = tuple(next(iter(channels.values()))["shape"][:2])

    IMPORTANCE_DIR = criteria_dir()
    scan_idx = state.get("scan_index", 0)                      # type: ignore
    dest = IMPORTANCE_DIR / f"scan_{scan_idx:02d}"
    dest.mkdir(parents=True, exist_ok=True)

    wd = _make_workdir(IMPORTANCE_DIR)
    backend = LocalBackend(wd, SANDBOX_PY, timeout=30)
    run_python = make_run_python(backend)
    model = ChatAnthropic(model=SEG_MODEL, temperature=0, max_tokens=SEG_MAX_TOKENS)  # type: ignore

    results = []
    for i, task in enumerate(tasks):
        for f in wd.glob("*"):                               # clean shared dir for this task
            if f.is_file():
                f.unlink()
        backend.archive_dir = dest / f"figures_{i}"          # per-task figure archive
        backend._step = 0

        agent = create_agent(model, tools=[run_python],
                             system_prompt=build_criteria_system_message(state.get("experiment_context")))

        # --- agent run + deliverable validation with retries ---
        base_msg = build_criteria_human_message(task, channels, seg)
        messages = [base_msg]
        for attempt in range(MAX_VALIDATION_RETRIES + 1):
            out = await agent.ainvoke(
                {"messages": messages}, # type: ignore
                config={"recursion_limit": SEG_MAX_SUPERSTEPS*2},
            )
            meta, errors = validate_criteria(wd / "labels.npy", wd / "criteria.json", grid_hw)
            if not errors:
                break

            messages = [base_msg, HumanMessage(                      # fresh start, no history
                "A previous attempt already ran in this working directory — its files are "
                "still there; re-load and reuse anything helpful (channels, intermediates). "
                "Its deliverables failed validation:\n- " + "\n- ".join(errors) +
                "\nProduce corrected labels.npy and criteria.json.")]
            print(errors)                                                                                                           #TEMP
            # messages = out["messages"] + [HumanMessage(
            #     "Your deliverables failed validation:\n- " + "\n- ".join(errors) +
            #     "\nFix the issues and re-save components.npy and components.json.")]
        if errors:
            raise RuntimeError(f"task {i} ({task!r}): criteria/labels invalid "
                               f"after {MAX_VALIDATION_RETRIES} retries: {errors}")

        # --- persist deliverables (scan- and task-indexed) ---
        lab_path  = dest / f"labels_{i}.npy"
        json_path = dest / f"criteria_{i}.json"
        code_path = dest / f"criteria_code_{i}.py"
        shutil.copy(wd / "labels.npy", lab_path); (wd / "labels.npy").unlink()
        shutil.copy(wd / "criteria.json", json_path); (wd / "criteria.json").unlink()
        code_path.write_text(extract_run_python_code(out["messages"]), encoding="utf-8")

        figures = sorted(str(p) for p in backend.archive_dir.glob("*.png"))

        results.append({
            "experiment_task": task,
            "labels_path": str(lab_path),
            "criteria_json_path": str(json_path),
            "criteria_meta": meta,
            "criteria_code_path": str(code_path),
            "reasoning": flatten_text(out["messages"][-1].content),
            "figures": figures,
        })
    return {"criteria_segmentations": results}