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


def build_backtest_diagnosis(metrics: dict) -> tuple[str, str]:
    """Simple qualitative interpretation of backtest metrics for UI."""
    sharpe = float(metrics.get("sharpe_ratio", 0))
    total_return = float(metrics.get("total_return", 0))
    max_dd = float(metrics.get("max_drawdown", 0))

    if sharpe >= 1.0 and total_return > 0 and max_dd > -0.25:
        return "✅ Выглядит устойчиво", "Профиль результата выглядит устойчивым для учебного проекта."
    if total_return > 0 and max_dd > -0.4:
        return "🟡 Приемлемо", "Есть прибыль, но риск/стабильность стоит улучшить (параметры, комиссии, walk-forward)."
    return "⚠️ Слабо / нестабильно", "Результат нестабилен: проверьте параметры, период, и устойчивость пары out-of-sample."


def build_method_comparison_text(coint_metrics: dict, corr_metrics: dict | None, best_method: str) -> str:
    """Human-readable interpretation for method benchmark section."""
    if corr_metrics is None:
        return "Корреляционный бенчмарк недоступен: сравнение выполнено только для коинтеграции."

    c_sharpe = float(coint_metrics.get("sharpe_ratio", 0))
    r_sharpe = float(corr_metrics.get("sharpe_ratio", 0))
    c_ret = float(coint_metrics.get("total_return", 0))
    r_ret = float(corr_metrics.get("total_return", 0))

    winner = "Коинтеграция" if best_method == "cointegration" else "Корреляция"
    return (
        f"**Лучший метод: {winner}.** "
        f"Шарп: {c_sharpe:.2f} vs {r_sharpe:.2f}; "
        f"Доходность: {c_ret:.2%} vs {r_ret:.2%}."
    )


def render_spread_chart(spread: pd.Series, zscore: pd.Series) -> go.Figure:
    """Render spread and z-score on two y-axes."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            mode="lines",
            name="Спред",
            line=dict(color="#1f77b4"),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=zscore.index,
            y=zscore.values,
            mode="lines",
            name="Z-оценка",
            line=dict(color="#ff7f0e"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Спред и Z-оценка",
        xaxis_title="Дата",
        yaxis=dict(title="Спред"),
        yaxis2=dict(title="Z-оценка", overlaying="y", side="right"),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickformat="%d.%m.%Y", hoverformat="%d.%m.%Y")
    return fig


def render_equity_chart(equity: pd.Series) -> go.Figure:
    """Render strategy equity curve."""
    fig = go.Figure(
        go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Капитал", line=dict(color="#2ca02c"))
    )
    fig.update_layout(title="Кривая капитала", xaxis_title="Дата", yaxis_title="Капитал", height=380)
    fig.update_xaxes(tickformat="%d.%m.%Y", hoverformat="%d.%m.%Y")
    return fig


def main() -> None:
    """Streamlit entrypoint."""
    st.set_page_config(page_title="Парный трейдинг MOEX", layout="wide")
    st.title("Поиск коинтегрированных пар (MOEX)")
    st.caption("Дипломный ИТ-проект: эконометрика как инструмент проектирования рыночно-нейтральной стратегии")

    with st.sidebar:
        st.header("Параметры")
        tickers = st.multiselect("Тикеры", options=data_config.tickers, default=data_config.tickers[:8])
        start_date = st.date_input("Дата начала", value=pd.to_datetime(data_config.start_date))
        end_date = st.date_input("Дата окончания", value=pd.to_datetime(data_config.end_date))
        missing_threshold = st.slider("Макс. доля пропусков", min_value=0.0, max_value=0.8, value=0.3, step=0.05)
        use_cache = st.checkbox("Использовать кэш MOEX", value=True)

        st.subheader("Коинтеграция")
        p_threshold = st.number_input(
            "Порог p-value",
            min_value=0.001,
            max_value=0.2,
            value=float(coint_config.p_value_threshold),
            step=0.001,
        )

        st.subheader("Стратегия")
        z_window = st.number_input(
            "Окно Z-оценки",
            min_value=5,
            max_value=120,
            value=int(strategy_config.zscore_window),
            step=1,
        )
        entry_z = st.number_input(
            "Порог входа (|Z|)",
            min_value=0.5,
            max_value=5.0,
            value=float(strategy_config.entry_z),
            step=0.1,
        )
        exit_z = st.number_input(
            "Порог выхода (|Z| <)",
            min_value=0.0,
            max_value=2.0,
            value=float(strategy_config.exit_z),
            step=0.1,
        )
        max_holding_days = st.number_input(
            "Макс. дней удержания",
            min_value=3,
            max_value=120,
            value=30,
            step=1,
        )

        run_btn = st.button("Запустить анализ", type="primary")

    if not run_btn:
        st.info("Задайте параметры в боковой панели и нажмите «Запустить анализ».")
        return

    if len(tickers) < 2:
        st.error("Выберите минимум 2 тикера.")
        return

    with st.spinner("Загружаем данные и выполняем анализ..."):
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

    st.subheader("Качество данных и предобработка")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Дней", int(quality.get("n_days", 0)))
    c2.metric("Тикеров", int(quality.get("n_tickers", 0)))
    c3.metric("Пропусков", int(quality.get("missing_values", 0)))
    c4.metric("Пропуски, %", f"{quality.get('missing_pct', 0):.2f}")
    st.dataframe(prices.tail(15), use_container_width=True)

    if result is None:
        st.warning("Для выбранного периода/настроек коинтегрированные пары не найдены.")
        return

    best_pair = result["best_pair"]
    signals = result["signals"]
    trades = result["trades"]
    metrics = result["metrics"]
    details = result["details"]
    corr_bt = result.get("correlation_backtest")
    comparison_table = pd.DataFrame(result.get("comparison_table", []))

    st.subheader("Лучшая коинтегрированная пара")
    st.markdown(
        f"**{best_pair['pair'][0]} - {best_pair['pair'][1]}**  \\\n"
        f"p-value: `{best_pair['p_value']:.6f}`, бета: `{best_pair['beta']:.4f}`, "
        f"R^2: `{best_pair['r_squared']:.4f}`, half-life: `{best_pair['half_life']:.1f}`"
    )

    left, right = st.columns(2)
    left.plotly_chart(render_spread_chart(best_pair["spread"], signals["zscore"]), use_container_width=True)
    right.plotly_chart(render_equity_chart(result["equity"]), use_container_width=True)

    st.subheader("Метрики бэктеста")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Совокупная доходность", f"{metrics['total_return']:.2%}")
    m2.metric("Годовая доходность", f"{metrics['annual_return']:.2%}")
    m3.metric("Коэф. Шарпа", f"{metrics['sharpe_ratio']:.2f}")
    m4.metric("Макс. просадка", f"{metrics['max_drawdown']:.2%}")

    st.markdown(f"**Оценочное число сделок:** {metrics['num_trades']}")
    st.markdown(f"**Доля прибыльных дней:** {metrics['win_rate']:.2%}")
    quality_title, quality_comment = build_backtest_diagnosis(metrics)
    st.info(f"**Качество бэктеста:** {quality_title}\n\n{quality_comment}")

    st.subheader("Интерпретация бэктеста (расширенная)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Дневная волатильность доходности", f"{details['volatility_daily']:.2%}")
    d2.metric("Лучший день (доходность)", f"{details['best_day']:.2%}")
    d3.metric("Худший день (доходность)", f"{details['worst_day']:.2%}")
    d4.metric("Доля прибыльных сделок", f"{details['trade_win_rate']:.2%}")
    st.caption("Здесь проценты относятся к дневной доходности стратегии, а не к длительности в днях.")
    st.markdown(
        f"- Среднее удержание: **{details['avg_holding_days']:.1f} дня**  \n"
        f"- Медианное удержание: **{details['median_holding_days']:.1f} дня**  \n"
        f"- Средний P&L сделки (в единицах спреда): **{details['avg_trade_pnl']:.4f}**"
    )

    st.subheader("Сравнение методов: коинтеграция vs корреляция")
    if corr_bt is None:
        st.warning("Корреляционный бенчмарк недоступен для выбранных настроек.")
    else:
        coint_metrics = metrics
        corr_metrics = corr_bt["metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Sharpe (коинт.)", f"{coint_metrics['sharpe_ratio']:.2f}")
        mc2.metric("Sharpe (корр.)", f"{corr_metrics['sharpe_ratio']:.2f}")
        mc3.metric("Доходность (коинт.)", f"{coint_metrics['total_return']:.2%}")
        mc4.metric("Доходность (корр.)", f"{corr_metrics['total_return']:.2%}")
        st.info(build_method_comparison_text(coint_metrics, corr_metrics, result["best_method"]))
        st.caption(result["comparison_reason"])

        corr_pair = corr_bt["pair"]
        st.markdown(
            f"Корреляционный эталон: **{corr_pair['pair'][0]} - {corr_pair['pair'][1]}**, "
            f"корр=`{corr_pair['correlation']:.4f}`, бета=`{corr_pair['beta']:.4f}`"
        )

    st.subheader("Матрица: корреляция vs коинтеграция")
    if comparison_table.empty:
        st.info("Матрица сравнения пуста.")
    else:
        st.dataframe(comparison_table.head(25), use_container_width=True)

    st.subheader("Сделки")
    if trades.empty:
        st.info("Для текущих настроек сделки не сгенерированы.")
    else:
        st.dataframe(trades, use_container_width=True)


if __name__ == "__main__":
    main()
