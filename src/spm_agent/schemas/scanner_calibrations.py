from pydantic import BaseModel, Field


class ScannerCalibrations(BaseModel, frozen=True):
    """
    One-point calibration relating the piezo scanner voltage (LVDT) coordinate
    system to the scan-frame (XOffset/YOffset) coordinate system.

    scanner_offset = lvdt_position - scan_center, both measured
    at the same physical point (tip parked at the scan-frame center by
    calibrate_xy_frame). Therefore:
        frame_coord   = lvdt_coord  - scanner_offset
        lvdt_coord    = frame_coord + scanner_offset

    Sensitivities are stored so sync can detect instrument
    reconfiguration by comparing against live XLVDTSens/YLVDTSens.
    Internal state - never rendered into LLM context.
    """
    x_scanner_offset_m: float
    y_scanner_offset_m: float
    x_lvdt_sens_m_per_v: float
    y_lvdt_sens_m_per_v: float


def to_scanner_calibration_state(state_dict: dict)-> ScannerCalibrations:
    scanner_calibration = ScannerCalibrations(
        x_scanner_offset_m = state_dict['x_lvdt_m'] - state_dict['x_scan_center_m'],
        y_scanner_offset_m = state_dict['y_lvdt_m'] - state_dict['y_scan_center_m'],
        x_lvdt_sens_m_per_v = state_dict['x_lvdt_sens_m_per_v'],
        y_lvdt_sens_m_per_v = state_dict['y_lvdt_sens_m_per_v']
    )

    return scanner_calibration