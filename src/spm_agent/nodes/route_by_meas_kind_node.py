from spm_agent.states.image_analysis_state import AnalysisState

def route_by_kind(state: AnalysisState) -> str:
    return state["kind"] 