def flatten_text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))

# def with_context(system_text: str, ctx=None) -> str:
#     if ctx is None:
#         return system_text
#     return system_text + "\n\n" + ctx.to_prompt_block()

def with_context(system_text: str, ctx: str | None = None) -> str:
    if not ctx:
        return system_text
    return system_text + "\n\nEXPERIMENT BACKGROUND\n" + ctx.strip()