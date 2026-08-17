"""Injectable time, sleeping, and randomness boundaries."""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class Sleeper(Protocol):
    async def sleep(self, seconds: float) -> None: ...


class RandomSource(Protocol):
    def uniform(self, lower: float, upper: float) -> float: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class AsyncioSleeper:
    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class SystemRandomSource:
    def __init__(self) -> None:
        self._random = random.SystemRandom()

    def uniform(self, lower: float, upper: float) -> float:
        return self._random.uniform(lower, upper)
