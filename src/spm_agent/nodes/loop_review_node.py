from langchain_anthropic import ChatAnthropic
from spm_agent.states.image_analysis_state import AnalysisState
from spm_agent.schemas.loop_review import LoopReview
from spm_agent.prompts.loop_review_prompt import build_loop_review_system_message, build_loop_review_human_message

from spm_agent.config import SEG_MODEL, SEG_MAX_TOKENS

async def loop_review_node(state: AnalysisState) -> AnalysisState:
    model = ChatAnthropic(model = SEG_MODEL,
                          temperature = 0,
                          max_tokens = SEG_MAX_TOKENS
                          )
    structured = model.with_structured_output(LoopReview)

    review = await structured.ainvoke(
        [build_loop_review_system_message(),
        build_loop_review_human_message(state["loop_params"]),] # type: ignore
    )

    return {"loop_review": review} #type: ignore