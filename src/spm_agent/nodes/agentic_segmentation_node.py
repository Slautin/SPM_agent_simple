from spm_agent.states.image_analysis_state import ImageAnalysisState
from spm_agent.config import SEG_MODEL, SEG_MAX_TOKENS, SEG_VISION_IN_LOOP, SEG_DIR, SEG_MAX_SUPERSTEPS
from spm_agent.tools.segmentation_toolbox import SegSession, build_segmentation_tools
from spm_agent.utils.channel_utils import load_array
from spm_agent.utils.image_utils import save_mask_and_overlay
from spm_agent.prompts.scan_segmentation_prompts import build_segmentation_human_message, build_segmentation_system_message

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent


async def agentic_segmentation_node(state: ImageAnalysisState) -> ImageAnalysisState:
    recs     = state["channel_recommendations"]["task_recommendation"] # pyright: ignore[reportTypedDictNotRequiredAccess]
    channels = state["file_channels"]  # type: ignore

    model   = ChatAnthropic(model = SEG_MODEL,
                            temperature=0,
                            max_tokens=SEG_MAX_TOKENS) # type: ignore
    
    results = {}
    for rec in recs[:2]:   #CHANGE IT LATER!!!
        if not rec["feasible"] or rec["primary_channel"] not in channels:
            continue
        task, channel_id = rec["task"], rec["primary_channel"]

        session = SegSession.from_raw(load_array(channels[channel_id].get("array_path")))
        agent   = create_agent(
            model,
            tools=build_segmentation_tools(session, SEG_VISION_IN_LOOP),
            system_prompt=build_segmentation_system_message(SEG_VISION_IN_LOOP),
        )

        out = await agent.ainvoke(
            {"messages": [build_segmentation_human_message(task, session)]},
            config={"recursion_limit": SEG_MAX_SUPERSTEPS},
        )

        # Claude returns content blocks → flatten to text
        final = out["messages"][-1].content
        final = final if isinstance(final, str) else " ".join(
            b.get("text", "") for b in final if isinstance(b, dict))
        
        mask_path, overlay_path = save_mask_and_overlay(session.view, session.mask, task, SEG_DIR)
        n_regions, coverage = session.mask_stats()

        results[task] = {
            "task": task, "channel": channel_id,
            "mask_path": mask_path, "overlay_path": overlay_path,
            "n_regions": n_regions, "coverage": round(coverage, 4),
            "ops": session.ops, "reasoning": final,
            "vision_in_loop": SEG_VISION_IN_LOOP, "model": SEG_MODEL,
            "n_look": sum(1 for o in session.ops if o["op"] == "show_overlay"),
        }

    return {"segmentation_results": results} # type: ignore
    
