"""Time helpers.

Everything in the orchestrator reads time through `now_ms()`, which is the asyncio
loop clock. In live mode that's real monotonic time. In the simulator we patch the
loop so it runs in *virtual time*: whenever the loop would block waiting for the next
timer, the clock jumps straight to it. A 2-minute simulated call finishes in
milliseconds, and the latency numbers are deterministic instead of being polluted
by OS scheduling jitter.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager


def now_ms() -> float:
    return asyncio.get_running_loop().time() * 1000.0


async def sleep_ms(ms: float) -> None:
    await asyncio.sleep(max(0.0, ms) / 1000.0)


async def sleep_until_ms(t_ms: float) -> None:
    await sleep_ms(t_ms - now_ms())


class VirtualClock:
    """Patch a SelectorEventLoop so time advances instantly to the next timer.

    Same idea as aiotools.VirtualClock. Only valid when the loop does no real I/O
    (the simulator). Blocking calls made inside a callback freeze virtual time,
    which is exactly what we want when replaying real network latencies (see
    deciders/bridge.py).
    """

    def __init__(self) -> None:
        self.t = 0.0

    @contextmanager
    def patch(self, loop: asyncio.AbstractEventLoop):
        selector = loop._selector  # type: ignore[attr-defined]
        orig_select = selector.select

        def vtime() -> float:
            return self.t

        def vselect(timeout=None):
            if timeout is not None and timeout > 0:
                self.t += timeout
            return orig_select(0)

        loop.time = vtime  # type: ignore[method-assign]
        selector.select = vselect
        try:
            yield self
        finally:
            selector.select = orig_select
            del loop.time  # restore the class method


def run_virtual(coro):
    """Run a coroutine to completion in virtual time (works on Windows too)."""
    loop = asyncio.SelectorEventLoop()
    clock = VirtualClock()
    try:
        asyncio.set_event_loop(loop)
        with clock.patch(loop):
            return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()
