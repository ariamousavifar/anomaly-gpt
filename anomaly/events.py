"""
Known market anomaly events registry.
Each event has a name, peak date, and a window for evaluation.
Sources: publicly documented market events with precise dates.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date


@dataclass
class MarketEvent:
    name:        str
    peak_date:   date          # single most significant day
    window_start: date         # evaluation window start
    window_end:   date         # evaluation window end
    description: str
    assets:      list[str]     # most affected assets


EVENTS: list[MarketEvent] = [
    MarketEvent(
        name         = "Flash Crash 2010",
        peak_date    = date(2010, 5, 6),
        window_start = date(2010, 5, 3),
        window_end   = date(2010, 5, 10),
        description  = "Dow drops ~1000 points intraday, recovers within minutes. "
                       "Triggered by algorithmic trading cascade.",
        assets       = ["SPY", "QQQ"],
    ),
    MarketEvent(
        name         = "China Devaluation Shock 2015",
        peak_date    = date(2015, 8, 24),
        window_start = date(2015, 8, 18),
        window_end   = date(2015, 9, 1),
        description  = "China devalues yuan, global equity selloff. "
                       "S&P 500 drops 11% in one week.",
        assets       = ["SPY", "QQQ", "GLD"],
    ),
    MarketEvent(
        name         = "XIV Vol Cascade 2018",
        peak_date    = date(2018, 2, 5),
        window_start = date(2018, 2, 1),
        window_end   = date(2018, 2, 12),
        description  = "VIX spikes 115% in one day, XIV ETN collapses. "
                       "S&P 500 loses ~10% in two weeks.",
        assets       = ["SPY", "VIX"],
    ),
    MarketEvent(
        name         = "COVID Crash 2020",
        peak_date    = date(2020, 3, 16),
        window_start = date(2020, 2, 20),
        window_end   = date(2020, 3, 23),
        description  = "Fastest bear market in history. "
                       "S&P 500 drops 34% in 33 days.",
        assets       = ["SPY", "QQQ", "VIX", "GLD", "TLT"],
    ),
    MarketEvent(
        name         = "Meme Stock Squeeze 2021",
        peak_date    = date(2021, 1, 27),
        window_start = date(2021, 1, 22),
        window_end   = date(2021, 2, 5),
        description  = "GameStop short squeeze, retail vs hedge funds. "
                       "Extreme volatility in broader market.",
        assets       = ["SPY", "QQQ", "VIX"],
    ),
    MarketEvent(
        name         = "Fed Rate Shock 2022",
        peak_date    = date(2022, 6, 13),
        window_start = date(2022, 6, 8),
        window_end   = date(2022, 6, 20),
        description  = "Hottest CPI print in 40 years triggers 75bps rate hike expectations. "
                       "S&P 500 enters bear market.",
        assets       = ["SPY", "QQQ", "TLT"],
    ),
    MarketEvent(
        name         = "SVB Collapse 2023",
        peak_date    = date(2023, 3, 10),
        window_start = date(2023, 3, 7),
        window_end   = date(2023, 3, 17),
        description  = "Silicon Valley Bank fails, largest US bank failure since 2008. "
                       "Banking contagion fears spread globally.",
        assets       = ["SPY", "TLT", "GLD"],
    ),
]

EVENT_NAMES = [e.name for e in EVENTS]


def get_event(name: str) -> MarketEvent:
    for e in EVENTS:
        if e.name == name:
            return e
    raise KeyError(f"Event '{name}' not found. Available: {EVENT_NAMES}")
