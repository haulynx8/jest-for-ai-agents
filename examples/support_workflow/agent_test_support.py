from jest_ai import expect, test_agent


@test_agent("handles refunds", tags=["support", "regression"])
async def handles_refunds(agent):
    response = await agent("I want a refund")
    expect(response).to_contain("refund")
