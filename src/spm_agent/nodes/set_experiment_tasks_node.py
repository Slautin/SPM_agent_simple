from spm_agent.states.image_analysis_state import AnalysisState

def make_experiment_tasks_node(experiment_tasks:list[str]):
    async def set_experiment_tasks_node(state: AnalysisState) -> AnalysisState:
        return {"experiment_tasks": experiment_tasks} # type: ignore
    return set_experiment_tasks_node