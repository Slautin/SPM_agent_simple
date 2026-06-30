from langchain_core.messages import SystemMessage, HumanMessage
from spm_agent.states.image_analysis_state import Channel
from spm_agent.utils.channel_utils import channel_to_image_block, channel_to_text_block


#channel recomendation node
CHANNEL_RECOMMENDATION_SYSTEM_PROMPT = (
        "You are an SPM/PFM channel-recommendation assistant.\n\n"
        "Your task is to recommend which of the available channels should be used "
        "for specific downstream image-analysis tasks. You will receive a main scientific task, "
        "channel metadata, basic channel statistics, and preview images.\n\n"

        "Recommend channels only for the following roles:\n"
        "1. ferroelectric domain segmentation\n"
        "2. ferroelectric domain wall segmentation\n"
        "3. grain boundary segmentation\n"
        "4. crack scratch detection\n"
        "5. surface contamination identification\n"
        "6. topography artifact check\n\n"

        "Rules:\n"
        "- Use only channels provided in the input.\n"
        "- Do not invent channel titles or channel IDs.\n"
        "- Do not perform segmentation.\n"
        "- Do not describe masks, coordinates, or objects as if they were already detected.\n"
        "- Base your recommendation on channel metadata, units, statistics, preview appearance, "
        "and the stated main scientific task.\n"
        "- A channel may be recommended for more than one role if justified.\n"
        "- If no channel is suitable for a role, return null or an empty list for that role and explain why.\n"
        "- If the recommendation is uncertain, lower the confidence and add a warning.\n\n"
        "Output requirements:\n"
        "- Return a structured recommendation only.\n"
        "- For each recommended channel, include the channel ID, role, confidence, and short reason.\n"
        "- Include an overall summary, warnings, and overall confidence.\n"
    )

def build_channel_recommendation_system_message() -> SystemMessage:
    return SystemMessage(content=CHANNEL_RECOMMENDATION_SYSTEM_PROMPT)


def build_channel_recommendation_human_message(file_channels: dict[str, Channel]) -> HumanMessage:
    main_task = {
        "type": "text",
        "text": (
            "Main task:\n"
            "Analyze SPM/PFM image data and select suitable channels for the analysis roles "
            "defined in the system instructions.\n\n"
            "Below are the available channels. For each channel, metadata is given first, "
            "followed immediately by the corresponding preview image."
        )
    }

    human_message_content = [main_task,]
    for ch in file_channels:
        text_message = channel_to_text_block(channel=file_channels[ch], channel_id=ch)
        human_message_content.append(text_message)
        image_message = channel_to_image_block(channel=file_channels[ch])
        human_message_content.append(image_message)


    human_message = HumanMessage(
        content= human_message_content, # pyright: ignore[reportArgumentType]
        ) # type: ignore
    
    return human_message