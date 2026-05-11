"""Streamlit interface for cointegrated-pairs search and backtesting."""

from __future__ import annotations

import os
import sys
import json
import logging
from importlib.util import find_spec

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import coint_config, data_config, strategy_config
from core.pipeline import load_and_prepare_data, prepare_experimental_data, run_full_pipeline
from strategy.backtest import calculate_backtest_metrics, calculate_equity_curve, plot_drawdown, plot_equity_curve
from strategy.signals import build_trades_table, generate_trading_signals, plot_spread_trades, plot_zscore_signals


logger = logging.getLogger(__name__)


def is_kaleido_available() -> bool:
    """Check whether kaleido is available for Plotly static image export."""
    return find_spec("kaleido") is not None


def safe_save_plotly_figure(fig: go.Figure, path: str) -> bool:
    """Safely save Plotly figure as PNG without breaking the app."""
    if not is_kaleido_available():
        logger.warning("PNG export skipped: kaleido is not installed (path=%s)", path)
        return False

    try:
        fig.write_image(path, width=1400, height=700)
        return True
    except Exception:
        logger.exception("Failed to save Plotly PNG to %s", path)
        return False


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
            value=int(strategy_config.max_holding_days),
            step=1,
        )

        selection_mode = st.radio(
            "Режим выбора пары",
            options=["auto", "manual"],
            format_func=lambda x: "Авто (min p-value)" if x == "auto" else "Ручной выбор",
        )
        run_btn = st.button("Запустить анализ", type="primary")

    if not run_btn:
        st.info("Задайте параметры в боковой панели и нажмите «Запустить анализ».")
        return

    if len(tickers) < 2:
        st.error("Выберите минимум 2 тикера.")
        return

    run_signature = {
        "tickers": tickers,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "missing_threshold": float(missing_threshold),
        "use_cache": bool(use_cache),
        "p_threshold": float(p_threshold),
        "z_window": int(z_window),
        "entry_z": float(entry_z),
        "exit_z": float(exit_z),
        "max_holding_days": int(max_holding_days),
    }
    signature_str = json.dumps(run_signature, sort_keys=True, ensure_ascii=False)
    if st.session_state.get("analysis_signature") != signature_str:
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("analysis_signature", None)
        st.session_state.pop("manual_pair", None)

    with st.spinner("Загружаем данные и выполняем анализ..."):
        try:
            experimental_data = prepare_experimental_data(
                tickers=tickers,
                start_date=str(start_date),
                end_date=str(end_date),
                use_cache=use_cache,
            )
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
            st.session_state["analysis_results"] = result
            st.session_state["analysis_signature"] = signature_str
        except Exception as exc:
            st.exception(exc)
            return

    st.subheader("Описание экспериментальных данных")
    quality_exp = experimental_data["quality"]
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Строк до очистки", quality_exp["Количество строк до очистки"])
    e2.metric("Строк после очистки", quality_exp["Количество строк после очистки"])
    e3.metric("Начало выборки", quality_exp["Дата начала выборки"])
    e4.metric("Окончание выборки", quality_exp["Дата окончания выборки"])

    st.markdown("**Пропущенные значения по каждому тикеру:**")
    missing_df = pd.DataFrame.from_dict(
        quality_exp["Пропуски по тикерам"], orient="index", columns=["Количество пропусков"]
    )
    missing_df.index.name = "Тикер"
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("**Первые строки подготовленной выборки цен закрытия:**")
    st.dataframe(experimental_data["head"], use_container_width=True)

    st.markdown("**Описательная статистика по инструментам:**")
    st.dataframe(experimental_data["stats"], use_container_width=True)

    st.markdown("**График динамики цен закрытия акций:**")
    st.image(experimental_data["files"]["plot_png"], caption="Динамика цен закрытия по выбранным тикерам")
    st.caption(
        "Результаты сохранены в файлы: "
        f"{experimental_data['files']['prices_csv']}, "
        f"{experimental_data['files']['stats_csv']}, "
        f"{experimental_data['files']['plot_png']}"
    )

    st.subheader("Качество данных и предобработка")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Дней", int(quality.get("n_days", 0)))
    c2.metric("Тикеров", int(quality.get("n_tickers", 0)))
    c3.metric("Пропусков", int(quality.get("missing_values", 0)))
    c4.metric("Пропуски, %", f"{quality.get('missing_pct', 0):.2f}")

    if result is None:
        st.warning("Для выбранного периода/настроек коинтегрированные пары не найдены.")
        st.dataframe(prices.tail(15), use_container_width=True)
        return

    coint_analysis = result.get("cointegration_analysis", {})
    results_df = coint_analysis.get("results_df", pd.DataFrame())
    pair_options = results_df["pair"].tolist() if not results_df.empty else []

    selected_pair = None
    if selection_mode == "manual":
        default_pair = st.session_state.get("manual_pair", pair_options[0] if pair_options else None)
        selected_pair_label = st.selectbox("Выберите пару вручную", options=pair_options, index=pair_options.index(default_pair) if default_pair in pair_options else 0)
        st.session_state["manual_pair"] = selected_pair_label
        left_ticker, right_ticker = selected_pair_label.split("-")
        selected_pair = (left_ticker, right_ticker)
        result = run_full_pipeline(
            prices=prices,
            p_value_threshold=p_threshold,
            z_window=int(z_window),
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            max_holding_days=int(max_holding_days),
            selected_pair=selected_pair,
            selection_mode="manual",
        )

    best_pair = result["best_pair"]
    signals = result["signals"]
    trades = result["trades"]
    metrics = result["metrics"]
    details = result["details"]
    corr_bt = result.get("correlation_backtest")
    comparison_table = pd.DataFrame(result.get("comparison_table", []))

    st.subheader("Таблицы: обработанные данные и коинтегрированные пары")
    tab_data, tab_pairs = st.tabs(["Обработанные данные", "Выявленная коинтеграция"])

    with tab_data:
        st.caption("Последние 15 строк обработанных и синхронизированных цен.")
        st.dataframe(prices.tail(15), use_container_width=True)

    with tab_pairs:
        if comparison_table.empty:
            st.info("Пары для сравнения не найдены.")
        else:
            coint_pairs = comparison_table.copy()

            if "is_cointegrated" in coint_pairs.columns:
                coint_pairs = coint_pairs[coint_pairs["is_cointegrated"] == True]  # noqa: E712

            if "status" in coint_pairs.columns:
                coint_pairs = coint_pairs[coint_pairs["status"].astype(str).str.contains("cointegr", case=False, na=False)]
            elif "p_value_coint" in coint_pairs.columns:
                coint_pairs = coint_pairs[coint_pairs["p_value_coint"].notna()]

            if coint_pairs.empty:
                st.info("Статистически коинтегрированные пары не найдены для выбранных параметров.")
            else:
                st.dataframe(coint_pairs, use_container_width=True)


    st.subheader("Анализ коинтеграционных зависимостей")
    coint_analysis = result.get("cointegration_analysis", {})
    results_df = coint_analysis.get("results_df", pd.DataFrame())
    saved_files = coint_analysis.get("saved_files", {})
    heatmap_path = coint_analysis.get("heatmap_path")

    total_pairs = len(results_df)
    coint_df = results_df[results_df["cointegrated"]].copy() if not results_df.empty else pd.DataFrame()
    coint_pairs_count = len(coint_df)

    k1, k2, k3 = st.columns(3)
    k1.metric("Проверено пар", total_pairs)
    k2.metric("Коинтегрированных пар", coint_pairs_count)
    if not results_df.empty:
        k3.metric("Режим выбора", "manual" if selection_mode == "manual" else "auto")

    st.markdown("**Полная таблица результатов (сортировка по p-value):**")
    st.dataframe(results_df, use_container_width=True)

    st.markdown(f"**Только коинтегрированные пары (p-value < {float(p_threshold):.3f}):**")
    if coint_df.empty:
        st.info("Коинтегрированные пары по выбранному критерию не найдены.")
    else:
        st.dataframe(coint_df, use_container_width=True)

    if heatmap_path:
        st.image(heatmap_path, caption="Тепловая карта p-value коинтеграционных зависимостей")
    if saved_files:
        st.caption(
            "Файлы раздела 3.2 сохранены: "
            f"{saved_files.get('full_csv', '-')}, {saved_files.get('cointegrated_csv', '-')}, {heatmap_path}"
        )

    st.subheader("Анализ динамики спреда")
    spread_analysis = result.get("spread_analysis", {})
    if spread_analysis:
        spread_pair = spread_analysis["pair"]
        spread_beta = spread_analysis["beta"]
        spread_df = spread_analysis["spread_df"]
        spread_stats = spread_analysis["spread_stats"]
        spread_files = spread_analysis["files"]

        st.markdown(f"**Выбранная пара:** {spread_pair[0]} - {spread_pair[1]}")
        st.markdown(f"**Коэффициент beta:** `{spread_beta:.4f}`")

        sa_col1, sa_col2 = st.columns(2)
        with sa_col1:
            st.image(spread_files["spread_plot_png"], caption="График динамики спреда")
        with sa_col2:
            st.image(spread_files["zscore_plot_png"], caption="График динамики Z-score")

        st.markdown("**Статистика спреда:**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Среднее", f"{spread_stats['mean']:.6f}")
        s2.metric("Ст. отклонение", f"{spread_stats['std']:.6f}")
        s3.metric("Минимум", f"{spread_stats['min']:.6f}")
        s4.metric("Максимум", f"{spread_stats['max']:.6f}")

        st.markdown(f"**Таблица spread и z-score (rolling={int(z_window)}):**")
        st.dataframe(spread_df[["spread", "z_score"]].dropna().tail(50), use_container_width=True)
        st.caption(
            "Файлы раздела 3.3 сохранены: "
            f"{spread_files['spread_zscore_csv']}, "
            f"{spread_files['spread_plot_png']}, "
            f"{spread_files['zscore_plot_png']}"
        )

    st.subheader("Лучшая коинтегрированная пара")
    st.markdown(
        f"**{best_pair['pair'][0]} - {best_pair['pair'][1]}**  \\\n"
        f"p-value: `{best_pair['p_value']:.6f}`, бета: `{best_pair['beta']:.4f}`, "
        f"R^2: `{best_pair['r_squared']:.4f}`, half-life: `{best_pair['half_life']:.1f}`"
    )
    debug_ctx = result.get("selection_context", {})
    st.caption(
        "DEBUG selection: "
        f"mode={debug_ctx.get('mode')}, "
        f"pair={debug_ctx.get('selected_pair')}, "
        f"source={debug_ctx.get('source')}, "
        f"p_value={best_pair['p_value']:.6f}, beta={best_pair['beta']:.4f}, "
        f"sample={debug_ctx.get('sample_start')}..{debug_ctx.get('sample_end')} (n={debug_ctx.get('sample_size')}), "
        f"params={debug_ctx.get('params')}"
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

    st.subheader("Сравнение коинтеграции и корреляции")
    comparison_section = result.get("comparison_section", {})
    corr_heatmap_path = comparison_section.get("correlation_heatmap_path")
    if corr_heatmap_path:
        st.image(corr_heatmap_path, caption="Heatmap корреляционной матрицы")

    st.subheader("Сравнение методов: коинтеграция vs корреляция")
    if corr_bt is None:
        st.warning("Корреляционный бенчмарк недоступен для выбранных настроек.")
    else:
        coint_metrics = metrics
        corr_metrics = corr_bt.get("metrics", {})
        if not corr_metrics:
            st.warning("Метрики корреляционного подхода не рассчитаны")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Sharpe (коинт.)", f"{coint_metrics['sharpe_ratio']:.2f}")
        mc2.metric("Sharpe (корр.)", f"{corr_metrics.get('sharpe_ratio', 0.0):.2f}")
        mc3.metric("Доходность (коинт.)", f"{coint_metrics['total_return']:.2%}")
        mc4.metric("Доходность (корр.)", f"{corr_metrics.get('total_return', 0.0):.2%}")
        st.info(build_method_comparison_text(coint_metrics, corr_metrics or None, result["best_method"]))
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

    # Глава 3.6: сопоставимые метрики и единые артефакты
    compare_rows = []
    coint_pair_name = f"{best_pair['pair'][0]}-{best_pair['pair'][1]}"
    compare_rows.append({
        "Подход": "Коинтеграция",
        "Пара": coint_pair_name,
        "Доходность": metrics["total_return"],
        "Sharpe": metrics["sharpe_ratio"],
        "Просадка": metrics["max_drawdown"],
        "Сделок": metrics["num_trades"],
        "Доля прибыльных сделок": metrics["win_rate"],
        "Итог": metrics.get("final_capital", 1.0),
    })

    if corr_bt is not None:
        corr_metrics = corr_bt.get("metrics", {})
        if not corr_metrics:
            st.warning("Метрики корреляционного подхода не рассчитаны")
        corr_pair_name = f"{corr_bt['pair']['pair'][0]}-{corr_bt['pair']['pair'][1]}"
        compare_rows.append({
            "Подход": "Корреляция",
            "Пара": corr_pair_name,
            "Доходность": corr_metrics.get("total_return", 0.0),
            "Sharpe": corr_metrics.get("sharpe_ratio", 0.0),
            "Просадка": corr_metrics.get("max_drawdown", 0.0),
            "Сделок": corr_metrics.get("num_trades", 0),
            "Доля прибыльных сделок": corr_metrics.get("win_rate", 0.0),
            "Итог": corr_metrics.get("final_capital", 1.0),
        })

    compare_df = pd.DataFrame(compare_rows)
    st.subheader("Сравнительная таблица (глава 3.6)")
    st.dataframe(compare_df, use_container_width=True)

    compare_chart = go.Figure()
    compare_chart.add_trace(go.Scatter(x=result["equity"].index, y=result["equity"].values, mode="lines", name="Коинтеграция"))
    if corr_bt is not None:
        compare_chart.add_trace(go.Scatter(x=corr_bt["equity"].index, y=corr_bt["equity"].values, mode="lines", name="Корреляция"))
    compare_chart.update_layout(title="Сравнение динамики капитала стратегий", xaxis_title="Дата", yaxis_title="Капитал")
    compare_chart.update_xaxes(showgrid=True)
    compare_chart.update_yaxes(showgrid=True)
    st.plotly_chart(compare_chart, use_container_width=True)

    st.caption("Пояснение: высокая корреляция не гарантирует устойчивую долгосрочную зависимость (коинтеграцию).")

    high_corr_not_coint = comparison_table[(comparison_table["correlation"].abs() >= 0.8) & (comparison_table["is_cointegrated"] == False)] if not comparison_table.empty else pd.DataFrame()
    if not high_corr_not_coint.empty:
        st.warning("Обнаружены пары с высокой корреляцией без коинтеграции — это потенциально ложные сигналы.")
        st.dataframe(high_corr_not_coint[["pair", "correlation", "p_value_coint", "status"]].head(10), use_container_width=True)

    output_dir = os.path.join("data", "results")
    os.makedirs(output_dir, exist_ok=True)
    comparison_csv = os.path.join(output_dir, "comparison_cointegration_vs_correlation_3_6.csv")
    comparison_png = os.path.join(output_dir, "capital_comparison_3_6.png")
    compare_df.to_csv(comparison_csv, index=False)
    cmp_saved = safe_save_plotly_figure(compare_chart, comparison_png)
    if not cmp_saved:
        st.warning("Не удалось сохранить PNG сравнительного графика капитала (3.6).")
    st.caption(f"Файлы раздела 3.6 сохранены: {comparison_csv}, {corr_heatmap_path}, {comparison_png}")

    st.subheader("3.4 Реализация торговой стратегии")
    st.markdown("Параметры стратегии задаются пользователем в панели слева и используются в расчётах без жёсткого кодирования.")

    strategy_signals = generate_trading_signals(
        spread=best_pair["spread"],
        rolling_window=int(z_window),
        entry_threshold=float(entry_z),
        exit_threshold=float(exit_z),
        max_holding_days=int(max_holding_days),
    )
    strategy_trades = build_trades_table(strategy_signals)

    zscore_fig = plot_zscore_signals(strategy_signals, float(entry_z), float(exit_z))
    spread_fig = plot_spread_trades(strategy_signals)

    zc1, zc2 = st.columns(2)
    zc1.plotly_chart(zscore_fig, use_container_width=True)
    zc2.plotly_chart(spread_fig, use_container_width=True)

    if strategy_trades.empty:
        st.info("Для выбранных параметров стратегии сделки не сформировались.")
    else:
        st.markdown("**Журнал сделок**")
        st.dataframe(strategy_trades[["entry_date", "exit_date", "position", "entry_z", "exit_z", "holding_days", "exit_reason", "pnl"]], use_container_width=True)

        total_trades = len(strategy_trades)
        profitable = int((strategy_trades["pnl"] > 0).sum())
        win_rate = profitable / total_trades if total_trades else 0.0
        avg_return = float(strategy_trades["pnl"].mean()) if total_trades else 0.0
        avg_holding = float(strategy_trades["holding_days"].mean()) if total_trades else 0.0

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Количество сделок", total_trades)
        t2.metric("Прибыльных", profitable)
        t3.metric("Доля прибыльных", f"{win_rate:.2%}")
        t4.metric("Средняя доходность сделки", f"{avg_return:.6f}")
        t5.metric("Средняя длительность", f"{avg_holding:.1f} дн.")

    output_dir = os.path.join("data", "results")
    os.makedirs(output_dir, exist_ok=True)
    trades_csv = os.path.join(output_dir, "trades_table_3_4.csv")
    zscore_png = os.path.join(output_dir, "zscore_signals_3_4.png")
    spread_png = os.path.join(output_dir, "spread_trades_3_4.png")
    strategy_trades.to_csv(trades_csv, index=False)

    kaleido_available = is_kaleido_available()
    zscore_saved = safe_save_plotly_figure(zscore_fig, zscore_png)
    spread_saved = safe_save_plotly_figure(spread_fig, spread_png)

    if not kaleido_available:
        st.warning("PNG-экспорт недоступен: не установлен пакет kaleido.")
    elif not (zscore_saved and spread_saved):
        st.warning("Не удалось сохранить часть PNG-графиков.")

    st.caption(f"Файлы раздела 3.4 сохранены: {trades_csv}, {zscore_png}, {spread_png}")

    st.subheader("3.5 Результаты бэктеста")
    bt1, bt2, bt3 = st.columns(3)
    initial_capital_input = bt1.number_input("Начальный капитал", min_value=1000.0, value=100000.0, step=1000.0)
    position_size_input = bt2.number_input("Доля капитала в сделке", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
    commission_input = bt3.number_input("Комиссия на сделку (доля)", min_value=0.0, max_value=1.0, value=0.0, step=0.0001)

    if strategy_trades.empty:
        st.warning("За выбранный период и параметры стратегия не сформировала сделок")
    else:
        equity_df = calculate_equity_curve(
            trades=strategy_trades,
            initial_capital=float(initial_capital_input),
            position_size=float(position_size_input),
            commission=float(commission_input),
        )
        bt_metrics = calculate_backtest_metrics(
            equity_df=equity_df,
            trades=strategy_trades,
            initial_capital=float(initial_capital_input),
        )

        metrics_table = pd.DataFrame(
            [
                ("Совокупная доходность", f"{bt_metrics['total_return']:.2%}"),
                ("Годовая доходность", f"{bt_metrics['annual_return']:.2%}"),
                ("Коэффициент Шарпа", f"{bt_metrics['sharpe_ratio']:.2f}"),
                ("Максимальная просадка", f"{bt_metrics['max_drawdown']:.2%}"),
                ("Количество сделок", bt_metrics["num_trades"]),
                ("Доля прибыльных сделок", f"{bt_metrics['win_rate']:.2%}"),
                ("Средняя доходность сделки", f"{bt_metrics['avg_trade_return']:.6f}"),
                ("Средняя длительность сделки", f"{bt_metrics['avg_holding_days']:.1f} дн."),
                ("Итоговый капитал", f"{bt_metrics['final_capital']:.2f}"),
            ],
            columns=["Показатель", "Значение"],
        )
        st.markdown("**Таблица показателей эффективности**")
        st.dataframe(metrics_table, use_container_width=True)

        equity_fig = plot_equity_curve(equity_df)
        drawdown_fig = plot_drawdown(equity_df)
        bc1, bc2 = st.columns(2)
        bc1.plotly_chart(equity_fig, use_container_width=True)
        bc2.plotly_chart(drawdown_fig, use_container_width=True)

        st.markdown("**Журнал сделок (раздел 3.4)**")
        st.dataframe(strategy_trades, use_container_width=True)

        metrics_csv = os.path.join(output_dir, "backtest_metrics_3_5.csv")
        trades_35_csv = os.path.join(output_dir, "trades_table_3_5.csv")
        equity_png = os.path.join(output_dir, "equity_curve_3_5.png")
        drawdown_png = os.path.join(output_dir, "drawdown_curve_3_5.png")

        metrics_table.to_csv(metrics_csv, index=False)
        strategy_trades.to_csv(trades_35_csv, index=False)
        equity_saved = safe_save_plotly_figure(equity_fig, equity_png)
        drawdown_saved = safe_save_plotly_figure(drawdown_fig, drawdown_png)

        if kaleido_available and not (equity_saved and drawdown_saved):
            st.warning("Не удалось сохранить часть PNG-графиков раздела 3.5.")

        st.caption(
            "Файлы раздела 3.5 сохранены: "
            f"{metrics_csv}, {trades_35_csv}, {equity_png}, {drawdown_png}"
        )


if __name__ == "__main__":
    main()
