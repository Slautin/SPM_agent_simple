from spm_agent.schemas.scanner_calibrations import ScannerCalibrations
from spm_agent.schemas.instrument import InstrumentState

from typing_extensions import TypedDict, NotRequired

class PFMExperimentState(TypedDict):
    """State of the PFM experiment, including the instrument state and the acquired data."""
    instrument_state: InstrumentState #current instrument parameter snapshot
    scanner_calibrations: ScannerCalibrations #define one time per session 