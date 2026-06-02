from langchain_openai import ChatOpenAI
from spm_agent.states.image_analysis_state import ImageAnalysisState
from spm_agent.schemas.channel_recommendation import TaskChannelReccomendationReport
from spm_agent.prompts.channel_recommendation_prompts import build_channel_recommendation_human_message, build_channel_recommendation_system_message


async def channel_recommendation_node(state: ImageAnalysisState) -> ImageAnalysisState:
    human_message = build_channel_recommendation_human_message(state["file_channels"]) # type: ignore
    system_message = build_channel_recommendation_system_message()
    model = ChatOpenAI(model="gpt-5.4",
                   temperature=0)
    
    structured_model = model.with_structured_output(TaskChannelReccomendationReport)
    
    report = await structured_model.ainvoke([
        system_message,
        human_message,
    ])
    
    return {
        "channel_recommendations": report.model_dump() # type: ignore
    } # type: ignore