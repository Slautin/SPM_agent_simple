from spm_agent.schemas.scanner_calibrations import ScannerCalibrations
from spm_agent.schemas.instrument import InstrumentState
from spm_agent.schemas.experimental_decision import ExperimentDecision
from spm_agent.schemas.scan_plan import ScanPlan

from spm_agent.states.image_analysis_state import AnalysisState

from typing_extensions import TypedDict, NotRequired
from typing import Literal

class PendingMeasurement(TypedDict):
    """The measurement currently being ordered/executed. Created by decide/pick,
    file_path filled by acquisition, consumed and cleared by analysis."""
    kind: Literal["scan", "loop"]              # intent — what was ordered
    decision_index: int                        # which decision ordered it
    params: NotRequired[ScanPlan]
    pixel_yx: NotRequired[tuple[int, int]]     # loops: where (from pick)
    file_path: NotRequired[str]                # set by acquisition
    

class ExperimentalRecord(AnalysisState):
    """One completed measurement + its analysis. `instrument_params` is a raw
    acquisition-time snapshot — field relevance depends on `kind`."""
    instrument_params: InstrumentState
    scan_index: int                            # scans: own number; loops: parent scan
    pixel_yx: NotRequired[tuple[int, int]]     # loops only, on that scan's grid

class SessionInfo(TypedDict):
    """Session-scoped facts read once at preflight, before the first tip motion.
    Internal - never rendered into LLM context."""
    run_dir: str
    started_ts: float                    # cutoff for "which .ibw is new"
    instrument_directory: str            # where the instrument writes .ibw files

class PFMExperimentState(TypedDict):
    """State of the PFM experiment."""
    session_info: SessionInfo

    instrument_state: InstrumentState                          # live snapshot always actual
    scanner_calibrations: ScannerCalibrations                  # once per session

    experiment_tasks: NotRequired[list[str]]
    experiment_context: NotRequired[str]

    experimental_records: NotRequired[list[ExperimentalRecord]]  # append-only history
    decision_records: NotRequired[list[ExperimentDecision]]
    pending: NotRequired[PendingMeasurement | None]

    
    #next_point_yx: NotRequired[tuple[int, int]] #not required, next point in px for the loop acquiring