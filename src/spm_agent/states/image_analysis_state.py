from typing_extensions import TypedDict, NotRequired

class ChannelStats(TypedDict):
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p99: float

class Channel(TypedDict):
    title: str
    units: str
    data_type: str
    shape: list
    preview_path: str
    array_path: str
    stats: ChannelStats

class SegmentationResult(TypedDict):
    task: str                 # joins back to channel_recommendations
    channel: str              # channel ID the agent worked on
    mask_path: str            # .npy boolean mask  (the pixels live here, not in state)
    overlay_path: str         # .png preview + red mask, for human/inspection
    n_regions: int            # connected components in the final mask
    coverage: float           # fraction of pixels masked (0..1)
    ops: list                 # ordered tool log = the reproducible "program"
    reasoning: str            # agent's final justification

class ImageAnalysisState(TypedDict):
    file_path: str
    file_channels: NotRequired[dict[str, Channel]]
    channel_recommendations: NotRequired[dict]
    segmentation_results: NotRequired[dict[str, SegmentationResult]] 
    pass