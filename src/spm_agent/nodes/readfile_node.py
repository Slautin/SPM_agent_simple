from spm_agent.mcp.scifireaders_service import SciFiReadersService
from spm_agent.states.image_analysis_state import ImageAnalysisState
from spm_agent.utils.image_utils import get_channel_stats, save_preview

import json


async def readfile_node(state: ImageAnalysisState) -> ImageAnalysisState:
    """
    Read an SPM file and prepare channel metadata, stats, and preview paths.
    """
    
    file_path = state["file_path"]
    
    channels = {}
    ch_keys = ['title', 'units', 'data_type', 'shape']

    #read channel
    service = SciFiReadersService()

    payload = await service.read_file(
        file_path,
    )



    payload_dict = payload['result']#json.loads(payload.content[0].text) # type: ignore

    #create dict
    for k in payload_dict['datasets']:
        channels[k] = {kk: payload_dict['datasets'][k][kk] for kk in ch_keys} 
        channels[k]['stats'] = get_channel_stats(payload_dict['datasets'][k]['data'])
        preview = save_preview(payload_dict['datasets'][k])

        if preview['ok']:
            channels[k]['preview_path'] = preview['path']
        else:
            print(preview['error'])

    return {
        'file_channels': channels
    } # type: ignore