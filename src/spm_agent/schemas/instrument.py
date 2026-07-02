from pydantic import BaseModel, Field
from spm_agent.schemas.scanner_calibrations import ScannerCalibrations



class ScanSettings(BaseModel, frozen=True):
    """Geometry and acquisition properties of the scan frame."""
    x_scan_center_m: float = Field(description="X center of the scan frame in meters")
    y_scan_center_m: float = Field(description="Y center of the scan frame in meters")
    scan_size_m: float = Field(description="Size (side length) of the scan frame in meters")
    pixels: int = Field(description="Number of pixels (per side) in the scan frame")
    scan_rate_hz: float = Field(description="Scan acquisition rate in Hz, i.e. number of lines per second")

class ProbePosition(BaseModel, frozen=True):
    x_m: float = Field(description="X position of the probe in meters, same coordinate system as the 'ScanSettings.x_scan_center_m' and 'ScanSettings.y_scan_center_m' fields")
    y_m: float = Field(description="Y position of the probe in meters, same coordinate system as the 'ScanSettings.x_scan_center_m' and 'ScanSettings.y_scan_center_m' fields")

class PFMExcitation(BaseModel, frozen=True):
    drive_amplitude_v: float = Field(description="AC excitation amplitude on the probe in volts")
    drive_frequency_hz: float = Field(description="Center frequency of the DART dual-frequency drive, Hz.")
    dart_width_hz: float = Field(description = "Width of the frequency window, separating the two drive frequencies, Hz. (f1,2 = drive_frequency_hz +/- dart_width_hz/2)")

class ContactFeedback(BaseModel, frozen=True):
    setpoint_v: float = Field(description="Deflection setpoint in volts, that determines the desired probe-sample contact force")
    gain: float = Field(description="Integral gain of the feedback loop controlling the probe height")

class InstrumentState(BaseModel, frozen=True):
    scan_settings: ScanSettings
    probe_position: ProbePosition
    pfm_excitation: PFMExcitation
    contact_feedback: ContactFeedback

def to_instrument_state(
        state_dict :dict,
        scanner_calibrations: ScannerCalibrations | None
        )-> InstrumentState:
    
    #probe position
    x_m, y_m = None, None
    
    x_probe_lvdt_m, y_probe_lvdt_m = state_dict.get("x_probe_lvdt_m"), state_dict.get("y_probe_lvdt_m")
    if scanner_calibrations is not None:
        x_m = x_probe_lvdt_m - scanner_calibrations.x_scanner_offset_m
        y_m = y_probe_lvdt_m - scanner_calibrations.y_scanner_offset_m

    probe_position = ProbePosition(
        x_m = x_m,
        y_m = y_m
    )
    
    scan_settings = ScanSettings(
        x_scan_center_m=state_dict["x_scan_center_m"],
        y_scan_center_m=state_dict["y_scan_center_m"],
        scan_size_m = state_dict['scan_size_m'],
        pixels = state_dict['n_points'],
        scan_rate_hz = state_dict['scan_rate_hz'],
    )

    pfm_excitation = PFMExcitation(
        drive_amplitude_v = state_dict["v_ac_v"],
        drive_frequency_hz = state_dict['f_dart_hz'],
        dart_width_hz = state_dict["f_dart_width_hz"],
    )

    contact_feedback = ContactFeedback(
        setpoint_v = state_dict['setpoint_defl_v'],
        gain = state_dict['igain']
    )
    
    return InstrumentState(
        probe_position = probe_position,
        scan_settings = scan_settings,
        pfm_excitation = pfm_excitation,
        contact_feedback = contact_feedback,
    )