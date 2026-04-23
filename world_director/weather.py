"""Markov-style weather transitions (rule-based)."""

from __future__ import annotations

from world_director.context_tokens import ContextToken

WEATHER_TRANSITIONS: dict[ContextToken, dict[ContextToken, float]] = {
    ContextToken.CLEAR_SKY: {
        ContextToken.CLEAR_SKY: 0.7,
        ContextToken.LIGHT_RAIN: 0.2,
        ContextToken.DUSK: 0.1,
    },
    ContextToken.LIGHT_RAIN: {
        ContextToken.LIGHT_RAIN: 0.5,
        ContextToken.HEAVY_RAIN: 0.3,
        ContextToken.CLEAR_SKY: 0.2,
    },
    ContextToken.HEAVY_RAIN: {
        ContextToken.HEAVY_RAIN: 0.4,
        ContextToken.THUNDERSTORM: 0.3,
        ContextToken.LIGHT_RAIN: 0.3,
    },
    ContextToken.THUNDERSTORM: {
        ContextToken.THUNDERSTORM: 0.3,
        ContextToken.HEAVY_RAIN: 0.5,
        ContextToken.CLEAR_SKY: 0.2,
    },
}
