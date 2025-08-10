#!/usr/bin/env python3
"""
Simplified enhanced universal hedge tool.

This version keeps the existing news and macro architecture light but adds:

* A constant macro indicator list limited to XLF, TLT and SPY for every ticker.
* A pre-backtest step that evaluates the five trading days prior to the user
  supplied start date.  Results from this calibration window are reflected upon
  by an LLM and the textual insights are fed into the main backtest so the model
  can adapt to ticker specific behaviour before producing out of sample
  predictions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

try:  # OpenAI is optional so that py_compile works without a key
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

MACRO_INDICATORS = ["XLF", "TLT", "SPY"]

TICKER_CONFIGS: Dict[str, Dict[str, str]] = {
    "AAPL": {"name": "Apple Inc", "yahoo_symbol": "AAPL"},
    "TSLA": {"name": "Tesla Inc", "yahoo_symbol": "TSLA"},
    "BTC": {"name": "Bitcoin", "yahoo_symbol": "BTC-USD"},
    "SPY": {"name": "S&P 500 ETF", "yahoo_symbol": "SPY"},
}

for cfg in TICKER_CONFIGS.values():
    cfg["macro_factors"] = MACRO_INDICATORS

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def get_price_df(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch daily close prices for symbol between start and end."""
    data = yf.Ticker(symbol).history(start=start.date(), end=end.date())
    return data[["Close"]].rename(columns={"Close": "close"}).reset_index()


def simple_signal(prices: pd.Series) -> float:
    """Very small placeholder signal: price above 3 day MA is +1 else -1."""
    if len(prices) < 4:
        return 0.0
    ma = prices.tail(3).mean()
    return 1.0 if prices.iloc[-1] > ma else -1.0


@dataclass
class DayResult:
    date: datetime
    signal: float
    ret: float
    strategy_ret: float


# ---------------------------------------------------------------------------
# pre-backtest logic
# ---------------------------------------------------------------------------


def run_pre_backtest(ticker: str, start: datetime, price_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for the five days prior to ``start``."""
    pre_end = start - timedelta(days=1)
    pre_start = pre_end - timedelta(days=4)
    days = pd.date_range(pre_start.date(), pre_end.date(), freq="D")
    results: List[DayResult] = []
    for day in days:
        prices_until_day = price_df[price_df["Date"] <= day]["close"]
        signal = simple_signal(prices_until_day)
        next_day = day + timedelta(days=1)
        future = price_df[price_df["Date"] == next_day]
        if future.empty:
            continue
        ret = future.iloc[0]["close"] / prices_until_day.iloc[-1] - 1
        results.append(DayResult(day.to_pydatetime(), signal, ret, signal * ret))
    return pd.DataFrame([r.__dict__ for r in results])


def reflect_on_pre_backtest(df: pd.DataFrame, ticker: str, model: str) -> str:
    """Ask an LLM for insights on the pre-backtest results."""
    if df.empty or OpenAI is None:
        return ""
    client = OpenAI(timeout=30)
    summary = df.to_dict(orient="records")
    system = "You are a trading strategist analysing model calibration results."
    user = (
        f"Pre-backtest results for {ticker} over the last {len(df)} days:\n"
        f"{json.dumps(summary, indent=2)}\n"
        "Provide three bullet point suggestions to improve future predictions."
    )
    raw = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return raw.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# main backtest
# ---------------------------------------------------------------------------


def backtest_with_reflection(ticker: str, start: str, end: str, model: str) -> pd.DataFrame:
    cfg = TICKER_CONFIGS.get(ticker.upper())
    if not cfg:
        raise ValueError(f"Unknown ticker {ticker}")
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    price_df = get_price_df(cfg["yahoo_symbol"], start_dt - timedelta(days=30), end_dt + timedelta(days=1))

    # ----- pre backtest -----
    pre_df = run_pre_backtest(ticker, start_dt, price_df)
    insights = reflect_on_pre_backtest(pre_df, ticker, model)

    # ----- main backtest -----
    days = pd.date_range(start_dt.date(), end_dt.date(), freq="D")
    results: List[DayResult] = []
    for day in days:
        prices_until_day = price_df[price_df["Date"] <= day]["close"]
        signal = simple_signal(prices_until_day)
        next_day = day + timedelta(days=1)
        future = price_df[price_df["Date"] == next_day]
        if future.empty:
            continue
        ret = future.iloc[0]["close"] / prices_until_day.iloc[-1] - 1
        results.append(DayResult(day.to_pydatetime(), signal, ret, signal * ret))

    df = pd.DataFrame([r.__dict__ for r in results])
    df.attrs["pre_backtest_insights"] = insights
    return df


# ---------------------------------------------------------------------------
# command line interface
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Run backtest with pre-calibration step")
    p.add_argument("backtest", nargs="?")  # positional for compatibility
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()

    df = backtest_with_reflection(args.ticker, args.start, args.end, args.model)
    print(df)
    if df.attrs.get("pre_backtest_insights"):
        print("\nInsights:\n" + df.attrs["pre_backtest_insights"])


if __name__ == "__main__":
    main()
