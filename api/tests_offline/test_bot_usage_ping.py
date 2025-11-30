from __future__ import annotations

import asyncio

from bot.commands.basic import (
    build_hyprl_ping_embed,
    build_predict_stats_embed,
    hyprl_ping_handler,
    hyprl_predict_stats_handler,
)


class DummyResponse:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def send_message(self, *args, **kwargs) -> None:
        self.calls.append({"args": args, "kwargs": kwargs})


class DummyInteraction:
    def __init__(self) -> None:
        self.response = DummyResponse()


class DummyClient:
    def __init__(self) -> None:
        self.api_base = "http://localhost:8000"
        self.request_count = 0

    async def get_usage(self) -> dict:
        self.request_count += 1
        return {
            "account_id": "bot_test",
            "credits_remaining": 1234,
            "by_endpoint": {"v2/predict": {"count": 10}},
        }

    async def get_predict_summary(self) -> dict:
        self.request_count += 1
        return {
            "total_predictions": 10,
            "closed_predictions": 6,
            "winrate_real": 0.5,
            "pnl_total": 4.0,
            "avg_pnl": 0.66,
        }


def test_hyprl_ping_handler_builds_embed() -> None:
    client = DummyClient()
    interaction = DummyInteraction()
    asyncio.run(hyprl_ping_handler(interaction, client))

    assert client.request_count == 1
    assert interaction.response.calls, "interaction should send a response"
    kwargs = interaction.response.calls[0]["kwargs"]
    assert kwargs.get("ephemeral") is True
    embed = kwargs.get("embed")
    assert embed is not None
    assert embed.fields[0].name == "Account"
    assert embed.fields[0].value == "bot_test"
    assert embed.fields[1].name == "Credits Remaining"
    assert "1234" in embed.fields[1].value


def test_build_hyprl_ping_embed_contents() -> None:
    payload = {"account_id": "acc", "credits_remaining": 42, "by_endpoint": {"v2/usage": {"count": 1}}}
    embed = build_hyprl_ping_embed("http://localhost:8000", payload)
    assert any(field.name == "API Base" and "localhost" in field.value for field in embed.fields)


def test_predict_stats_handler_and_embed() -> None:
    client = DummyClient()
    interaction = DummyInteraction()
    asyncio.run(hyprl_predict_stats_handler(interaction, client))

    assert client.request_count == 1
    kwargs = interaction.response.calls[0]["kwargs"]
    assert kwargs.get("ephemeral") is True
    embed = kwargs.get("embed")
    assert embed is not None
    assert any(field.name == "Total Predictions" for field in embed.fields)
    stats_embed = build_predict_stats_embed(
        {"total_predictions": 5, "closed_predictions": 3, "winrate_real": 0.4, "pnl_total": 2.0, "avg_pnl": 0.4}
    )
    assert any("0.4" in field.value for field in stats_embed.fields)
