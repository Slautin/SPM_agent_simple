from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from spm_agent.config import SEG_MODEL, SEG_MAX_TOKENS, decisions_dir
from spm_agent.schemas.experimental_decision import ExperimentDecision
from spm_agent.states.pfm_experiment_state import PFMExperimentState
from spm_agent.utils.decision_utils import build_decision_digest
from spm_agent.prompts.decision_prompt import build_decision_system_message
from spm_agent.config import MAX_TOTAL_DECISIONS

def _guard_decision(action, frame_action, why):
    return ExperimentDecision(
        understanding="(deterministic guard)",
        open_questions="",
        action=action,
        frame_action=frame_action,
        target_criterion=None,
        reasoning=why)

async def experiment_decision_node(state: PFMExperimentState) -> PFMExperimentState:
    recs      = state.get("experimental_records", [])
    decisions = state.get("decision_records", [])

    if state.get("pending") is not None:
        raise RuntimeError(f"decide called with a measurement still in flight: {state['pending']}")

    if not any(r["kind"] == "image" for r in recs):          # no scan yet -> must scan
        decision = _guard_decision("scan", "hold", "No image measured yet; a scan is required first.")
    elif len(decisions) >= MAX_TOTAL_DECISIONS:              # runaway protection
        decision = _guard_decision("stop", None, f"Safety cap: {MAX_TOTAL_DECISIONS} decisions reached.")
    else:
        work_dir = decisions_dir() / f"decision_{len(decisions):02d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        digest = build_decision_digest(state)
        (work_dir / "digest.txt").write_text(digest, encoding="utf-8")

        model = ChatAnthropic(model=SEG_MODEL, temperature=0, max_tokens=SEG_MAX_TOKENS)  # type: ignore
        decision = await model.with_structured_output(ExperimentDecision).ainvoke(
            [build_decision_system_message(state.get("experiment_context")),
             HumanMessage(content=digest)])
        (work_dir / "decision.json").write_text(
            decision.model_dump_json(indent=2), encoding="utf-8")# type: ignore
        
    pending = (None if decision.action == "stop" else {"kind": decision.action, "decision_index": len(decisions)})

    return {"decision_records": decisions + [decision], "pending": pending}      # type: ignore