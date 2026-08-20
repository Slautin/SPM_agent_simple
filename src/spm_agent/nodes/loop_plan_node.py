from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from spm_agent.config import SEG_MODEL, SEG_MAX_TOKENS, decisions_dir
from spm_agent.prompts.loop_plan_prompt import (
    build_loop_plan_system_message, build_loop_plan_human_message)
from spm_agent.schemas.loop_plan import LoopPlan, validate_loop_plan, loop_plan_diff
from spm_agent.states.pfm_experiment_state import PFMExperimentState

MAX_PLAN_RETRIES = 2

def _archive(plan: LoopPlan, decision_index: int) -> None:
    """Lands beside the decision that ordered it: decision.json, pick.json, loop_plan.json."""
    d = decisions_dir() / f"decision_{decision_index:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "loop_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

async def loop_plan_node(state: PFMExperimentState) -> PFMExperimentState:
    """Choose the switching waveform for the next hysteresis loop. The location is
    already fixed by pick_point; this node sets only how the site is measured."""
    pending = state["pending"]
    if pending["kind"] != "loop":                                     # type: ignore
        raise RuntimeError(f"loop_plan called for kind={pending['kind']!r}")   # type: ignore
    if "pixel_yx" not in pending:                                     # type: ignore
        raise RuntimeError("loop_plan requires pixel_yx — pick_point must run first")

    live = state.get("instrument_state")
    if live is None:
        raise RuntimeError("loop_plan requires instrument_state — run sync_status first")

    recs = state.get("experimental_records", [])
    idx  = pending["decision_index"]                                  # type: ignore

    # first loop of the session: no loop outcome to diagnose from
    if not any(r.get("kind") == "loop" for r in recs):
        plan = LoopPlan(
            diagnosis="(deterministic guard) first loop of the session",
            loop_settings=live.loop_settings,
            reasoning="No prior loop exists to tune against; adopt the operator's "
                      "waveform unchanged.")
        _archive(plan, idx)
        print(f"[loop_plan] first loop — operator waveform adopted "
              f"(v_dc_max {live.loop_settings.v_dc_max_v:.2f} V, "
              f"{live.loop_settings.n_cycles} cycles)")
        return {"pending": {**pending, "params": plan}}                # type: ignore

    system = build_loop_plan_system_message(state.get("experiment_context"))
    human  = build_loop_plan_human_message(state)
    structured = ChatAnthropic(
        model=SEG_MODEL, temperature=0,
        max_tokens=SEG_MAX_TOKENS).with_structured_output(LoopPlan)    # type: ignore

    messages = [system, human]
    plan, errors = None, ["no plan produced"]
    for _ in range(MAX_PLAN_RETRIES + 1):
        plan = await structured.ainvoke(messages)
        errors = validate_loop_plan(plan, live)                        # type: ignore
        if not errors:
            break
        print(f"[loop_plan] rejected: {errors}")
        messages = [system, human, HumanMessage(                       # fresh, no history
            "Your plan was rejected:\n- " + "\n- ".join(errors) +
            "\nReturn a corrected waveform, restating every field.")]
    if errors:
        raise RuntimeError(f"loop plan invalid after {MAX_PLAN_RETRIES} retries: {errors}")

    diff = loop_plan_diff(plan, live)                                  # type: ignore
    print(f"[loop_plan] {plan.diagnosis}")                             # type: ignore
    print(f"[loop_plan] changes: {diff or 'none — repeats current waveform'}")

    _archive(plan, idx)                                                # type: ignore
    return {"pending": {**pending, "params": plan}}                    # type: ignore


