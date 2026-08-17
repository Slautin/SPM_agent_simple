from langchain_anthropic import ChatAnthropic
from spm_agent.config import CHAN_MODEL, CHAN_MAX_TOKENS, channels_dir
from spm_agent.utils.io_utils import save_json

from spm_agent.states.image_analysis_state import AnalysisState
from spm_agent.schemas.channel_recommendation import TaskChannelReccomendationReport
from spm_agent.prompts.channel_recommendation_prompts import build_channel_recommendation_human_message, build_channel_recommendation_system_message


async def channel_recommendation_node(state: AnalysisState) -> AnalysisState:
    human_message = build_channel_recommendation_human_message(state["file_channels"]) # type: ignore
    system_message = build_channel_recommendation_system_message(state.get("experiment_context"))
    model = ChatAnthropic(model=CHAN_MODEL, temperature=0, max_tokens=CHAN_MAX_TOKENS)  # type: ignore
    
    structured_model = model.with_structured_output(TaskChannelReccomendationReport)
    
    report = await structured_model.ainvoke([
        system_message,
        human_message,
    ])

    scan_idx = state.get("scan_index", 0)                                    # type: ignore
    out_path = save_json(
        channels_dir() / f"scan_{scan_idx:02d}" / "channel_recommendation.json",
        {"scan_index": scan_idx, "model": CHAN_MODEL, "report": report})

    return {"channel_recommendations": report,                               # type: ignore
            "channel_recommendations_path": str(out_path)}                   # type: ignore
