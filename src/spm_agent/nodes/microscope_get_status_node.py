from spm_agent.mcp.spm_mcp_client import SPMMCPClient
from spm_agent.schemas.instrument import to_instrument_state
from spm_agent.states.pfm_experiment_state import PFMExperimentState

import json

async def get_status_node(state: PFMExperimentState) -> PFMExperimentState:
    spm_client = SPMMCPClient()
    result = await spm_client.call_tool("pfm_get_experiment_status")
    res_dict = json.loads(result[0]['text'])

    if res_dict['ok']:
        state_dict = res_dict['data']
        instrument_state = to_instrument_state(
            state_dict,
            scanner_calibrations=state['scanner_calibrations']
            )
    
    return {
        "instrument_state": instrument_state
    }