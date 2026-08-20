from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from spm_agent.config import SUMMARY_MAX_TOKENS, SUMMARY_MODEL, run_dir
from spm_agent.prompts.summary_prompt import build_summary_system_message
from spm_agent.schemas.experiment_summary import ExperimentSummary, validate_summary
from spm_agent.states.pfm_experiment_state import PFMExperimentState
from spm_agent.utils.decision_utils import build_summary_digest

MAX_SUMMARY_RETRIES = 2


async def summary_node(state: PFMExperimentState) -> PFMExperimentState:
    """Close the campaign: answer the tasks from everything measured. Terminal —
    orders no measurement and touches no instrument."""
    recs      = state.get("experimental_records", [])
    tasks     = state.get("experiment_tasks", [])

    if not any(r["kind"] == "image" for r in recs):
        raise RuntimeError("summary requires at least one analysed scan")

    digest = build_summary_digest(state)
    d = run_dir() / "summary"
    d.mkdir(parents=True, exist_ok=True)
    (d / "digest.txt").write_text(digest, encoding="utf-8")

    system = build_summary_system_message(state.get("experiment_context"))
    structured = ChatAnthropic(
        model=SUMMARY_MODEL, temperature=0,
        max_tokens=SUMMARY_MAX_TOKENS).with_structured_output(ExperimentSummary)  # type: ignore

    messages = [system, HumanMessage(content=digest)]
    summary, errors = None, ["no summary produced"]
    for _ in range(MAX_SUMMARY_RETRIES + 1):
        summary = await structured.ainvoke(messages)
        errors = validate_summary(summary, tasks)                          # type: ignore
        if not errors:
            break
        print(f"[summary] rejected: {errors}")
        messages = [system, HumanMessage(content=digest),                  # fresh, no history
                    HumanMessage("Your summary was rejected:\n- " + "\n- ".join(errors)
                                 + "\nReturn a corrected summary covering exactly the "
                                   "tasks listed, using their exact wording.")]
    if errors:
        raise RuntimeError(f"summary invalid after {MAX_SUMMARY_RETRIES} retries: {errors}")

    (d / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8")               # type: ignore

    for f in summary.task_findings:                                        # type: ignore
        print(f"[summary] {f.task}\n    {f.answer}  (confidence: {f.confidence})")
        
    return {"experiment_summary": summary}                                 # type: ignore