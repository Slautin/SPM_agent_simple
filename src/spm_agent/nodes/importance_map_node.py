from spm_agent.config import importance_dir #IMPORTANCE_DIR
from spm_agent.tools.run_python import make_run_python, LocalBackend, extract_run_python_code
from spm_agent.utils.message_utils import flatten_text
from spm_agent.utils.importance_map_utils import build_map
from spm_agent.config import SANDBOX_PY, SEG_MODEL, SEG_MAX_TOKENS, SEG_MAX_SUPERSTEPS
from spm_agent.prompts.importance_map_prompts import build_importance_human_message, build_importance_system_message
from spm_agent.schemas.importance_components import validate_components, validate_safety
from spm_agent.states.image_analysis_state import AnalysisState

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from pathlib import Path
import shutil
import numpy as np
import json

MAX_VALIDATION_RETRIES = 2



def _make_workdir(importance_dir) -> Path:
    wd = importance_dir / "work"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    return wd


async def importance_map_node(state: AnalysisState) -> AnalysisState:
    tasks    = state.get("experiment_tasks", [])
    channels = state["file_channels"]           # type: ignore
    recs     = state.get("channel_recommendations")      # type: ignore
    grid_hw  = tuple(next(iter(channels.values()))["shape"][:2])

    IMPORTANCE_DIR = importance_dir()
    scan_idx = state.get("scan_index", 0)                      # type: ignore
    dest = IMPORTANCE_DIR / f"scan_{scan_idx:02d}"
    dest.mkdir(parents=True, exist_ok=True)

    wd = _make_workdir(IMPORTANCE_DIR)
    backend = LocalBackend(wd, SANDBOX_PY, timeout=30)
    run_python = make_run_python(backend)
    model = ChatAnthropic(model=SEG_MODEL, temperature=0, max_tokens=SEG_MAX_TOKENS)  # type: ignore

    importance_maps = []
    for i, task in enumerate(tasks):
        for f in wd.glob("*"):                               # clean shared dir for this task
            if f.is_file():
                f.unlink()
        backend.archive_dir = dest / f"figures_{i}"          # per-task figure archive
        backend._step = 0

        agent = create_agent(model, tools=[run_python],
                             system_prompt=build_importance_system_message(state.get("experiment_context")))

        # --- agent run + deliverable validation with retries ---
        base_msg = build_importance_human_message(
            task, channels, recs,
            )

        messages = [base_msg]
        for attempt in range(MAX_VALIDATION_RETRIES + 1):
            out = await agent.ainvoke(
                {"messages": messages}, # type: ignore
                config={"recursion_limit": SEG_MAX_SUPERSTEPS*2},
            )
            meta, errors, warnings = validate_components(
                wd / "components.npy", wd / "components.json", grid_hw)
            if not errors:
                comps = np.load(wd / "components.npy")
                s_meta, s_err, s_warn = validate_safety(
                    wd / "safety.npy", wd / "safety.json", grid_hw, components=comps)
                errors, warnings = s_err, list(warnings) + list(s_warn)
                if not errors:
                    break

            messages = [base_msg, HumanMessage(              # fresh start, files still on disk
                "A previous attempt already ran in this working directory — its files are "
                "still there; re-load and reuse anything helpful. Its deliverables failed "
                "validation:\n- " + "\n- ".join(errors) +
                 "\nProduce corrected components.npy / components.json / "
                "safety.npy / safety.json.")]
            
        if errors:
            raise RuntimeError(f"task {i} ({task!r}): deliverables invalid after "
                               f"{MAX_VALIDATION_RETRIES} retries: {errors}")

        # --- persist deliverables (scan- and task-indexed) ---
        comp_path   = dest / f"components_{i}.npy"
        json_path   = dest / f"components_{i}.json"
        safety_npy  = dest / f"safety_{i}.npy"
        safety_json = dest / f"safety_{i}.json"
        task_path   = dest / f"task_map_{i}.npy"
        map_path    = dest / f"importance_map_{i}.npy"
        code_path   = dest / f"scoring_code_{i}.py"

        shutil.move(wd / "components.npy",  comp_path)
        shutil.move(wd / "components.json", json_path)
        shutil.move(wd / "safety.npy",      safety_npy)
        shutil.move(wd / "safety.json",     safety_json)

        (dest / f"validation_{i}.json").write_text(
            json.dumps({"warnings": warnings}, indent=2, ensure_ascii=False), encoding="utf-8")

        # --- deterministic maps: OUR arithmetic, not the agent's ---
        task_map = build_map(np.load(comp_path))
        safety   = np.load(safety_npy)
        np.save(task_path, task_map)
        np.save(map_path,  task_map * safety)

        code_path.write_text(extract_run_python_code(out["messages"]), encoding="utf-8")

        importance_maps.append({
            "experiment_task":      task,
            "components_path":      str(comp_path),
            "components_json_path": str(json_path),
            "components_meta":      meta,
            "safety_map_path":      str(safety_npy),
            "safety_json_path":     str(safety_json),
            "safety_meta":          s_meta,
            "task_map_path":        str(task_path),
            "importance_map_path":  str(map_path),   # digest + no-criterion fallback only
            "scoring_code_path":    str(code_path),
            "reasoning":            flatten_text(out["messages"][-1].content),
            "figures":              sorted(str(p) for p in backend.archive_dir.glob("*.png")),
            "warnings":             warnings,
        })

    return {"importance_maps": importance_maps}    # type: ignore                        