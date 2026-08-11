from spm_agent.states.image_analysis_state import AnalysisState
from langgraph.graph import StateGraph, START, END

from spm_agent.nodes.agentic_segmentation_node import agentic_segmentation_node
from spm_agent.nodes.channel_recommendation_node import channel_recommendation_node
from spm_agent.nodes.set_experiment_tasks_node import make_experiment_tasks_node
from spm_agent.nodes.importance_map_node import importance_map_node
from spm_agent.nodes.build_loop_node import build_loop_node
from spm_agent.nodes.readfile_node import readfile_node
from spm_agent.nodes.route_by_meas_kind_node import route_by_kind
from spm_agent.nodes.loop_params_node import loop_params_node
from spm_agent.nodes.loop_review_node import loop_review_node

def build_analysis_graph():
    # --- image subgraph: the whole existing chain, compiled separately ---
    img = StateGraph(AnalysisState)
    img.add_node("channel_recommendation", channel_recommendation_node)
    img.add_node("segmentation", agentic_segmentation_node)
    img.add_node("importance_map", importance_map_node)
    img.add_edge(START, "channel_recommendation")
    img.add_edge("channel_recommendation", "segmentation")  
    img.add_edge("segmentation", "importance_map")
    img.add_edge("importance_map", END)

    image_graph = img.compile()

    # --- loop subgraph, same idea ---
    lp = StateGraph(AnalysisState)
    lp.add_node("build_loop", build_loop_node)
    lp.add_node("loop_params", loop_params_node)
    lp.add_node("loop_review", loop_review_node)

    lp.add_edge(START, "build_loop")
    lp.add_edge("build_loop", "loop_params")
    lp.add_edge("loop_params", "loop_review")
    lp.add_edge("loop_review", END)
    loop_graph = lp.compile()

    # --- parent analysis graph ---
    builder = StateGraph(AnalysisState)
    builder.add_node("readfile", readfile_node)
    builder.add_node("image_analysis", image_graph)     # compiled graph as node
    builder.add_node("loop_analysis", loop_graph)

    builder.add_edge(START, "readfile")
    builder.add_conditional_edges("readfile", route_by_kind,
                                {"image": "image_analysis",
                                "loop": "loop_analysis",
                                "spectrum": END})


    builder.add_edge("image_analysis", END)
    builder.add_edge("loop_analysis", END)

    return builder.compile()