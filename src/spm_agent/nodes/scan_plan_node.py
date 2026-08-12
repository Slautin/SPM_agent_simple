# src/spm_agent/nodes/scan_plan_node.py
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from spm_agent.config import SEG_MODEL, SEG_MAX_TOKENS, decisions_dir
from spm_agent.prompts.scan_plan_prompt import (
    build_scan_plan_system_message, build_scan_plan_human_message)
from spm_agent.schemas.scan_plan import (
    ScanPlan, validate_plan, plan_diff, locked_fields)
from spm_agent.states.pfm_experiment_state import PFMExperimentState

MAX_PLAN_RETRIES = 2


def _archive(plan: ScanPlan, decision_index: int) -> None:
    """Plan lands next to the decision that ordered it: digest.txt, decision.json,
    scan_plan.json — the whole 'why, then how' of one cycle in one folder."""
    d = decisions_dir() / f"decision_{decision_index:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scan_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

async def scan_plan_node(state: PFMExperimentState) -> PFMExperimentState:
    """Choose HOW WELL to measure the next scan — and, when frame_action allows it,
    how large a frame. Never where: the centre is computed, never proposed."""
    pending = state["pending"]
    if pending["kind"] != "scan":                                        # type: ignore
        raise RuntimeError(
            f"scan_plan_node called for kind={pending['kind']!r}")       # type: ignore

    live = state.get("instrument_state")
    if live is None:
        raise RuntimeError("scan_plan_node requires instrument_state — run sync_status first")

    decision = state["decision_records"][-1]
    recs     = state.get("experimental_records", [])

    if not any(r.get("kind") == "image" for r in recs):
        plan = ScanPlan(
            diagnosis="(deterministic guard) first scan of the session",
            scan_settings=live.scan_settings,
            pfm_excitation=live.pfm_excitation,
            reasoning="No prior scan exists to tune against; adopt the operator's "
                      "configuration unchanged.")
        _archive(plan, pending["decision_index"])                        # type: ignore
        print("[scan_plan] first scan — operator settings adopted")
        return {"pending": {**pending, "params": plan}}                  # type: ignore

    lock_scan, lock_exc = locked_fields(decision.frame_action)
    system = build_scan_plan_system_message(state.get("experiment_context"))
    human  = build_scan_plan_human_message(state, locked=lock_scan + lock_exc)
    structured = ChatAnthropic(
        model=SEG_MODEL, temperature=0,
        max_tokens=SEG_MAX_TOKENS).with_structured_output(ScanPlan)      # type: ignore

    messages = [system, human]
    plan, errors = None, ["no plan produced"]
    for _ in range(MAX_PLAN_RETRIES + 1):
        plan = await structured.ainvoke(messages)
        errors = validate_plan(plan, live, decision.frame_action)        # type: ignore
        if not errors:
            break
        print(f"[scan_plan] rejected: {errors}")
        messages = [system, human, HumanMessage(                          # fresh, no history
            "Your plan was rejected:\n- " + "\n- ".join(errors) +
            "\nReturn a corrected plan, copying the locked fields unchanged.")]
    if errors:
        raise RuntimeError(f"scan plan invalid after {MAX_PLAN_RETRIES} retries: {errors}")

    diff = plan_diff(plan, live)                                          # type: ignore
    print(f"[scan_plan] {plan.diagnosis}")                                # type: ignore
    print(f"[scan_plan] changes: {diff or 'none — repeats current settings'}")

    _archive(plan, pending["decision_index"])                             # type: ignore
    return {"pending": {**pending, "params": plan}}                       # type: ignore