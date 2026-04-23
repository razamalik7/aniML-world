"""Context token IDs — multi-hot appended to observations (ecosystem phase)."""

from __future__ import annotations

from enum import IntEnum


class ContextToken(IntEnum):
    CLEAR_SKY = 0
    LIGHT_RAIN = 1
    HEAVY_RAIN = 2
    THUNDERSTORM = 3
    METEOR_INCOMING = 4
    METEOR_IMPACT = 5
    NIGHT = 6
    DAWN = 7
    DUSK = 8
    DROUGHT = 9
    WILDFIRE = 10


NUM_CONTEXT_TOKENS = len(ContextToken)


def encode_context(active_tokens: list[ContextToken]) -> list[float]:
    vec = [0.0] * NUM_CONTEXT_TOKENS
    for t in active_tokens:
        vec[int(t)] = 1.0
    return vec
