import asyncio


async def support_agent(prompt: str) -> str:
    """A tiny agent used by the MVP examples and test suite."""
    await asyncio.sleep(0.1)
    if "refund" in prompt.lower():
        return "I can help you initiate a refund."
    return "How can I help you today?"
