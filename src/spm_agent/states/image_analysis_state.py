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
    stats: ChannelStats

class ImageAnalysisState(TypedDict):
    #input
    file_path: str

    file_channels: NotRequired[dict[str, Channel]]

    channel_recommendations: NotRequired[dict]
    pass