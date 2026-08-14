from spm_agent.mcp.scifireaders_service import SciFiReadersService
from spm_agent.states.image_analysis_state import AnalysisState

from spm_agent.utils.channel_utils import (save_array, save_preview, get_stats, fix_units,
                                           save_channel_grid)
from spm_agent.config import channels_dir

import json
import numpy as np
from spm_agent.utils.loops_utils import classify_measurement

import os

ORIENT_K = 1



async def readfile_node(state: AnalysisState) -> AnalysisState:
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
    payload_dict = payload['result']
    for k, ds in payload_dict['datasets'].items():
        ds['units'] = fix_units(ds['title'], ds['units'])  #temporary due to the bug inn SCFIReaders

    for ds in payload_dict['datasets'].values():
        a = np.asarray(ds['data'], dtype=np.float32)
        if a.ndim == 2 and min(a.shape) > 1:
            a = np.ascontiguousarray(np.rot90(a, k=ORIENT_K))
        ds['data']  = a
        ds['shape'] = list(a.shape)
    # -----------------------------------------------------------------------

    #create dict
    for k in payload_dict['datasets']:
        channels[k] = {kk: payload_dict['datasets'][k][kk] for kk in ch_keys} 
        channels[k]['stats'] = get_stats(payload_dict['datasets'][k]['data'])

        preview = save_preview(payload_dict['datasets'][k])
        if preview['ok']:
            channels[k]['preview_path'] = preview['path']
        else:
            print(preview['error'])

        dat = save_array(payload_dict['datasets'][k])  
        if dat['ok']:
            channels[k]['array_path'] = dat['path']
        else:
            print(dat['error'])

    #classify
    arrays = {ds['title']: np.asarray(ds['data'], dtype=np.float32)
          for ds in payload_dict['datasets'].values()}
    
    measurement_kind = classify_measurement(arrays)

    out = {'file_channels': channels, 
           'kind': measurement_kind}

    if measurement_kind == "image":                       # loop/spectrum files are 1-D
        scan_idx = state.get("scan_index", 0)             # type: ignore
        grid = save_channel_grid(
            channels, channels_dir() / f"scan_{scan_idx:02d}" / "preview_grid.png")
        if grid["ok"]:
            out['preview_grid_path'] = grid["path"]
        else:
            print(grid["error"])

    return out  # type: ignore

    