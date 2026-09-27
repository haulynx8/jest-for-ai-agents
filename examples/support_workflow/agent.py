import asyncio


async def support_agent(prompt: str) -> str:
    """A tiny agent used by the executable workflow example."""
    await asyncio.sleep(0.01)
    if "refund" in prompt.casefold():
        return "I can help initiate a refund."
    return "How can I help you today?"
