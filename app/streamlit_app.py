"""Streamlit interface for cointegrated-pairs search and backtesting."""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import coint_config, data_config, strategy_config
from core.pipeline import load_and_prepare_data, run_full_pipeline


def render_spread_chart(spread: pd.Series, zscore: pd.Series) -> go.Figure:
    """Render spread and z-score on two y-axes."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            mode="lines",
            name="Spread",
            line=dict(color="#1f77b4"),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=zscore.index,
            y=zscore.values,
            mode="lines",
            name="Z-score",
            line=dict(color="#ff7f0e"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Spread and Z-score",
        xaxis_title="Date",
        yaxis=dict(title="Spread"),
        yaxis2=dict(title="Z-score", overlaying="y", side="right"),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_equity_chart(equity: pd.Series) -> go.Figure:
    """Render strategy equity curve."""
    fig = go.Figure(
        go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity", line=dict(color="#2ca02c"))
    )
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Capital", height=380)
    return fig


def main() -> None:
    """Streamlit entrypoint."""
    st.set_page_config(page_title="Pairs Trading MOEX", layout="wide")
    st.title("Cointegrated Pairs Discovery (MOEX)")
    st.caption("IT diploma project: econometrics as a tool for market-neutral strategy design")

    with st.sidebar:
        st.header("Parameters")
        tickers = st.multiselect("Tickers", options=data_config.tickers, default=data_config.tickers[:8])
        start_date = st.date_input("Start date", value=pd.to_datetime(data_config.start_date))
        end_date = st.date_input("End date", value=pd.to_datetime(data_config.end_date))
        missing_threshold = st.slider("Max missing share", min_value=0.0, max_value=0.8, value=0.3, step=0.05)
        use_cache = st.checkbox("Use MOEX cache", value=True)

        st.subheader("Cointegration")
        p_threshold = st.number_input(
            "p-value threshold",
            min_value=0.001,
            max_value=0.2,
            value=float(coint_config.p_value_threshold),
            step=0.001,
        )

        st.subheader("Strategy")
        z_window = st.number_input(
            "Z-score window",
            min_value=5,
            max_value=120,
            value=int(strategy_config.zscore_window),
            step=1,
        )
        entry_z = st.number_input(
            "Entry threshold (|Z|)",
            min_value=0.5,
            max_value=5.0,
            value=float(strategy_config.entry_z),
            step=0.1,
        )
        exit_z = st.number_input(
            "Exit threshold (|Z| <)",
            min_value=0.0,
            max_value=2.0,
            value=float(strategy_config.exit_z),
            step=0.1,
        )
        max_holding_days = st.number_input(
            "Max holding days",
            min_value=3,
            max_value=120,
            value=30,
            step=1,
        )

        run_btn = st.button("Run analysis", type="primary")

    if not run_btn:
        st.info("Set parameters in the sidebar and click 'Run analysis'.")
        return

    if len(tickers) < 2:
        st.error("Please select at least 2 tickers.")
        return

    with st.spinner("Loading data and running analysis..."):
        try:
            prices, quality = load_and_prepare_data(
                tickers=tickers,
                start_date=str(start_date),
                end_date=str(end_date),
                missing_threshold=missing_threshold,
                use_cache=use_cache,
            )
            result = run_full_pipeline(
                prices=prices,
                p_value_threshold=p_threshold,
                z_window=int(z_window),
                entry_z=float(entry_z),
                exit_z=float(exit_z),
                max_holding_days=int(max_holding_days),
            )
        except Exception as exc:
            st.exception(exc)
            return

    st.subheader("Data quality and preprocessing")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days", int(quality.get("n_days", 0)))
    c2.metric("Tickers", int(quality.get("n_tickers", 0)))
    c3.metric("Missing values", int(quality.get("missing_values", 0)))
    c4.metric("Missing, %", f"{quality.get('missing_pct', 0):.2f}")
    st.dataframe(prices.tail(15), use_container_width=True)

    if result is None:
        st.warning("No cointegrated pairs found for selected period/settings.")
        return

    best_pair = result["best_pair"]
    signals = result["signals"]
    trades = result["trades"]
    metrics = result["metrics"]

    st.subheader("Best cointegrated pair")
    st.markdown(
        f"**{best_pair['pair'][0]} - {best_pair['pair'][1]}**  \\\n"
        f"p-value: `{best_pair['p_value']:.6f}`, beta: `{best_pair['beta']:.4f}`, "
        f"R^2: `{best_pair['r_squared']:.4f}`, half-life: `{best_pair['half_life']:.1f}`"
    )

    left, right = st.columns(2)
    left.plotly_chart(render_spread_chart(best_pair["spread"], signals["zscore"]), use_container_width=True)
    right.plotly_chart(render_equity_chart(result["equity"]), use_container_width=True)

    st.subheader("Backtest metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Return", f"{metrics['total_return']:.2%}")
    m2.metric("Annual Return", f"{metrics['annual_return']:.2%}")
    m3.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    m4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

    st.markdown(f"**Estimated trades count:** {metrics['num_trades']}")
    st.markdown(f"**Profitable days share:** {metrics['win_rate']:.2%}")

    st.subheader("Trades")
    if trades.empty:
        st.info("No trades generated for current settings.")
    else:
        st.dataframe(trades, use_container_width=True)


if __name__ == "__main__":
    main()