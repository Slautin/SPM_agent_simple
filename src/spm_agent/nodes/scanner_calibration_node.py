from spm_agent.mcp.spm_mcp_client import SPMMCPClient
from spm_agent.schemas.scanner_calibrations import to_scanner_calibration_state
from spm_agent.states.pfm_experiment_state import PFMExperimentState

import json


async def get_scanner_calibrations_node(state: PFMExperimentState) -> PFMExperimentState:
    spm_client = SPMMCPClient()
    result = await spm_client.call_tool("pfm_calibrate_xy_frame")
    res_dict = json.loads(result[0]['text'])

    if res_dict['ok']:
        state_dict = res_dict['data']
        scanner_calibrations = to_scanner_calibration_state(state_dict)
    
    return {
        "scanner_calibrations": scanner_calibrations
    }