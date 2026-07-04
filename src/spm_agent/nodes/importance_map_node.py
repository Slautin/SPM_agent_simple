from spm_agent.config import IMPORTANCE_DIR
from spm_agent.tools.run_python import make_run_python, LocalBackend, extract_run_python_code
from spm_agent.utils.message_utils import flatten_text
from spm_agent.config import SANDBOX_PY, SEG_MODEL, SEG_MAX_TOKENS, SEG_MAX_SUPERSTEPS
from spm_agent.prompts.importance_map_prompts import build_importance_human_message, build_importance_system_message
from spm_agent.states.image_analysis_state import ImageAnalysisState

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from pathlib import Path
import shutil

def _make_workdir() -> Path:
    wd = IMPORTANCE_DIR / "work"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    return wd

async def importance_map_node(state: ImageAnalysisState) -> ImageAnalysisState:
    tasks    = state.get("experiment_tasks", [])
    channels = state["file_channels"]           # type: ignore
    seg      = state["segmentation_results"]     # type: ignore

    wd = _make_workdir()
    backend = LocalBackend(wd, SANDBOX_PY, timeout=30)
    run_python = make_run_python(backend)
    model = ChatAnthropic(model=SEG_MODEL, temperature=0, max_tokens=SEG_MAX_TOKENS)  # type: ignore
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

    importance_maps = []
    for i, task in enumerate(tasks):
        for f in wd.glob("*"):                               # clean shared dir for this task
            if f.is_file():
                f.unlink()
        backend.archive_dir = IMPORTANCE_DIR / f"figures_{i}"  # per-task figure archive
        backend._step = 0

        agent = create_agent(model, tools=[run_python],
                             system_prompt=build_importance_system_message())

        out = await agent.ainvoke(
            {"messages": [build_importance_human_message(task, channels, seg)]},
            config={"recursion_limit": SEG_MAX_SUPERSTEPS*2},
        )

        src       = wd / "importance_map.npy"
        map_path  = IMPORTANCE_DIR / f"importance_map_{i}.npy"
        code_path = IMPORTANCE_DIR / f"scoring_code_{i}.py"
        if src.exists():
            shutil.copy(src, map_path)
            src.unlink()
        code_path.write_text(extract_run_python_code(out["messages"]))

        figures = sorted(str(p) for p in backend.archive_dir.glob("*.png"))   # NEW

        importance_maps.append({
            "experiment_task": task,
            "importance_map_path": str(map_path),
            "scoring_code_path": str(code_path),
            "reasoning": flatten_text(out["messages"][-1].content),
            "candidate_regions": [],           # filled later by the deterministic pick node
        })

    
    return {"importance_maps": importance_maps}   # type: ignore   