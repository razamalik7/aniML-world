"""World Director orchestration — time, weather, catastrophe hooks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from world_director.context_tokens import ContextToken, encode_context
from world_director.weather import WEATHER_TRANSITIONS


@dataclass
class WorldDirector:
    """Rule-based director; ``step`` returns an 11D multi-hot context vector."""

    transition_interval_s: float = 60.0
    meteor_probability_per_hour: float = 0.01
    weather_state: ContextToken = ContextToken.CLEAR_SKY
    time_of_day_hours: float = 8.0
    day_length_s: float = 600.0
    _tokens: set[ContextToken] = field(default_factory=set)
    _weather_timer: float = 0.0

    def step(self, dt: float, ecosystem_state: dict | None = None) -> list[float]:
        ecosystem_state = ecosystem_state or {}
        self._advance_time(dt)
        self._weather_timer += dt
        if self._weather_timer >= self.transition_interval_s:
            self._weather_timer = 0.0
            self._update_weather()
        self._check_catastrophes(ecosystem_state)
        return encode_context(list(self._tokens))

    def _advance_time(self, dt: float) -> None:
        if self.day_length_s <= 0.0:
            return
        delta_hours = 24.0 * (dt / self.day_length_s)
        self.time_of_day_hours = (self.time_of_day_hours + delta_hours) % 24.0
        self._sync_day_night_tokens()

    def _sync_day_night_tokens(self) -> None:
        for t in (ContextToken.DAWN, ContextToken.DUSK, ContextToken.NIGHT):
            self._tokens.discard(t)
        h = self.time_of_day_hours
        if 5.0 <= h < 7.0:
            self._tokens.add(ContextToken.DAWN)
        elif 18.0 <= h < 20.0:
            self._tokens.add(ContextToken.DUSK)
        elif h < 5.0 or h >= 20.0:
            self._tokens.add(ContextToken.NIGHT)

    def _update_weather(self) -> None:
        table = WEATHER_TRANSITIONS.get(self.weather_state)
        if not table:
            return
        roll = random.random()
        acc = 0.0
        for nxt, p in table.items():
            acc += p
            if roll <= acc:
                self.weather_state = nxt
                break
        for w in (
            ContextToken.CLEAR_SKY,
            ContextToken.LIGHT_RAIN,
            ContextToken.HEAVY_RAIN,
            ContextToken.THUNDERSTORM,
        ):
            self._tokens.discard(w)
        if self.weather_state in (
            ContextToken.CLEAR_SKY,
            ContextToken.LIGHT_RAIN,
            ContextToken.HEAVY_RAIN,
            ContextToken.THUNDERSTORM,
        ):
            self._tokens.add(self.weather_state)

    def _check_catastrophes(self, ecosystem_state: dict) -> None:
        p = self.meteor_probability_per_hour * (1.0 / 3600.0)
        if random.random() < p:
            self._tokens.add(ContextToken.METEOR_INCOMING)
        herb = float(ecosystem_state.get("herbivore_capacity_ratio", 0.0))
        if herb > 0.8:
            self._tokens.add(ContextToken.DROUGHT)
        elif herb < 0.5:
            self._tokens.discard(ContextToken.DROUGHT)
