"""Engine registry. Order here is the order shown in the app."""
from .adaptive_learner import AdaptiveLearnerEngine
from .repnet_torch import RepNetTorchEngine
from .hand_landmarks import HandLandmarkEngine
from .optical_flow import OpticalFlowEngine
from .motion_energy import MotionEnergyEngine
from .repnet_tf_legacy import RepNetTFLegacyEngine
from .transrac_official import TransRACEngine

ENGINES = {e.id: e for e in [
    AdaptiveLearnerEngine(),
    RepNetTorchEngine(),
    HandLandmarkEngine(),
    OpticalFlowEngine(),
    MotionEnergyEngine(),
    TransRACEngine(),
    RepNetTFLegacyEngine(),
]}



def engine_status():
    """{engine_id: (available, message)} - never raises."""
    out = {}
    for eid, e in ENGINES.items():
        try:
            out[eid] = e.check()
        except Exception as ex:   # a broken install must not break the app
            out[eid] = (False, f"Check failed: {ex}")
    return out
