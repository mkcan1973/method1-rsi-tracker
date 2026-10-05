"""Instrument universe for method1, mirroring the Spec pattern from seasonal/specs.py."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    ticker: str          # yfinance symbol
    name: str
    group: str           # "equity", "sector", "rate", "commodity", "macro"
    tradeable: bool = True  # False for context-only series like VIX


UNIVERSE: dict[str, Spec] = {
    spec.ticker: spec
    for spec in [
        # Broad equity
        Spec("SPY", "S&P 500 ETF", "equity"),
        Spec("QQQ", "Nasdaq 100 ETF", "equity"),
        Spec("IWM", "Russell 2000 ETF", "equity"),
        Spec("DIA", "Dow Jones ETF", "equity"),
        Spec("^GSPC", "S&P 500 Index", "equity"),

        # Sectors
        Spec("XLF", "Financials", "sector"),
        Spec("XLE", "Energy", "sector"),
        Spec("XLK", "Technology", "sector"),
        Spec("XLY", "Consumer Discretionary", "sector"),
        Spec("XLP", "Consumer Staples", "sector"),
        Spec("XLV", "Health Care", "sector"),
        Spec("XLI", "Industrials", "sector"),

        # Rates
        Spec("TLT", "20+ Year Treasury", "rate"),
        Spec("IEF", "7-10 Year Treasury", "rate"),
        Spec("SHY", "1-3 Year Treasury", "rate"),

        # Commodity
        Spec("GLD", "Gold", "commodity"),

        # Macro context only (not traded on directly)
        Spec("^VIX", "CBOE Volatility Index", "macro", tradeable=False),
    ]
}


def tradeable_tickers() -> list[str]:
    return [t for t, s in UNIVERSE.items() if s.tradeable]


def all_tickers() -> list[str]:
    return list(UNIVERSE.keys())
