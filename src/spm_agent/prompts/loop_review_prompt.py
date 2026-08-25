from langchain_core.messages import SystemMessage, HumanMessage
from spm_agent.utils.image_utils import image_path_to_data_url
import json

LOOP_REVIEW_SYSTEM_PROMPT = (
    "You are a PFM spectroscopy reviewer.\n\n"
    "You receive: (1) hysteresis parameters extracted deterministically from an "
    "SS-PFM measurement, and (2) annotated plots of the off-field and on-field "
    "loops with the extracted values drawn on top.\n\n"
    "Your tasks:\n"
    "- Verify the numbers against the images\n"
    "- Classify loop quality and determine whether the observed response can be reliably "
    "attributed to true ferroelectric switching or is more consistent with electrostatic, "
    "noise-related, or other artifacts.\n"
    "- Give a short physical interpretation.\n\n"
    "Rules: base every claim on the provided data; list concrete issues."
)

def build_loop_review_system_message() -> SystemMessage:
    return SystemMessage(content=LOOP_REVIEW_SYSTEM_PROMPT)

def build_loop_review_human_message(loop_params: dict) -> HumanMessage:
    content = []
    for branch, params in loop_params.items():
        p = params.model_dump()
        img = p.pop("annotated_path")
        content.append({"type": "text",
                        "text": f"--- {branch}-field loop parameters ---\n"
                                + json.dumps(p, indent=1)})
        if img:
            content.append({"type": "image_url",
                            "image_url": {"url": image_path_to_data_url(img)}})
    return HumanMessage(content=content)