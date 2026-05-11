"""Streamlit interface for cointegrated-pairs search and backtesting."""
from __future__ import annotations

import json
import logging
import os
import sys
from importlib.util import find_spec

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import coint_config, data_config, strategy_config
from core.pipeline import load_and_prepare_data, prepare_experimental_data, run_full_pipeline

logger = logging.getLogger(__name__)


def is_kaleido_available() -> bool:
    return find_spec("kaleido") is not None


def safe_save_plotly_figure(fig: go.Figure, path: str) -> bool:
    if not is_kaleido_available():
        logger.warning("PNG export skipped: kaleido is not installed (path=%s)", path)
        return False
    try:
        fig.write_image(path, width=1400, height=700)
        return True
    except Exception:
        logger.exception("Failed to save Plotly PNG to %s", path)
        return False


def render_equity_chart(equity: pd.Series, title: str = "Кривая капитала") -> go.Figure:
    is_index = len(equity) > 0 and float(equity.iloc[0]) == 1.0
    y_title = "Индекс капитала" if is_index else "Капитал"
    fig = go.Figure(go.Scatter(x=equity.index, y=equity.values, mode="lines", name=y_title))
    fig.update_layout(title=title, xaxis_title="Дата", yaxis_title=y_title, height=380)
    return fig


def build_method_comparison_text(coint_metrics: dict, corr_metrics: dict | None) -> str:
    if corr_metrics is None:
        return "Корреляционный бенчмарк недоступен: сравнение выполнено только для коинтеграции."
    c_ret = float(coint_metrics.get("total_return", 0))
    r_ret = float(corr_metrics.get("total_return", 0))
    c_sh = float(coint_metrics.get("sharpe_ratio", 0))
    r_sh = float(corr_metrics.get("sharpe_ratio", 0))
    base = f"Сравнение результатов при выбранных параметрах: Sharpe {c_sh:.2f} vs {r_sh:.2f}; доходность {c_ret:.2%} vs {r_ret:.2%}."
    if c_ret > 0 or r_ret > 0:
        winner = "Коинтеграция" if c_ret >= r_ret else "Корреляция"
        return f"{base} Лидирует: {winner}."
    return base


def main() -> None:
    st.set_page_config(page_title="Парный трейдинг MOEX", layout="wide")
    st.title("Экспериментальная глава: парный трейдинг MOEX")

    with st.sidebar:
        st.header("Параметры")
        tickers = st.multiselect("Тикеры", data_config.tickers, default=data_config.tickers[:8])
        start_date = st.date_input("Дата начала", value=pd.to_datetime(data_config.start_date))
        end_date = st.date_input("Дата окончания", value=pd.to_datetime(data_config.end_date))
        missing_threshold = st.slider("Макс. доля пропусков", 0.0, 0.8, 0.3, 0.05)
        use_cache = st.checkbox("Использовать кэш MOEX", value=True)

        st.subheader("Коинтеграция и стратегия")
        p_threshold = st.number_input("Порог p-value", 0.001, 0.2, float(coint_config.p_value_threshold), 0.001)
        z_window = st.number_input("Окно Z-оценки", 5, 120, int(strategy_config.zscore_window), 1)
        entry_z = st.number_input("Порог входа |Z|", 0.5, 5.0, float(strategy_config.entry_z), 0.1)
        exit_z = st.number_input("Порог выхода |Z|<", 0.0, 2.0, float(strategy_config.exit_z), 0.1)
        max_holding_days = st.number_input("Макс. дней удержания", 3, 120, int(strategy_config.max_holding_days), 1)

        st.subheader("Фильтры качества пары (опционально)")
        use_filters = st.checkbox("Применять фильтры в авто-выборе", value=False)
        min_r2 = st.number_input("Минимальный R²", 0.0, 1.0, 0.0, 0.05)
        min_abs_beta = st.number_input("Минимальный |beta|", 0.0, 10.0, 0.0, 0.1)
        min_half_life = st.number_input("Half-life от", 0.0, 500.0, 0.0, 1.0)
        max_half_life = st.number_input("Half-life до", 1.0, 500.0, 500.0, 1.0)

        selection_mode = st.radio("Режим выбора пары", ["auto", "manual"], format_func=lambda x: "Авто" if x == "auto" else "Ручной")
        run_btn = st.button("Запустить анализ", type="primary")

    if not run_btn:
        st.info("Нажмите «Запустить анализ».")
        return
    if len(tickers) < 2:
        st.error("Выберите минимум 2 тикера.")
        return

    signature = json.dumps({"tickers": tickers, "start": str(start_date), "end": str(end_date), "selection_mode": selection_mode, "p": p_threshold, "z": z_window, "e": entry_z, "x": exit_z, "h": max_holding_days, "filters": [use_filters, min_r2, min_abs_beta, min_half_life, max_half_life]}, sort_keys=True)
    if st.session_state.get("analysis_signature") != signature:
        st.session_state.pop("analysis_results", None)
        st.session_state["analysis_signature"] = signature

    experimental_data = prepare_experimental_data(tickers, str(start_date), str(end_date), use_cache)
    prices, quality = load_and_prepare_data(tickers, str(start_date), str(end_date), missing_threshold, use_cache)
    base_result = run_full_pipeline(
        prices=prices,
        p_value_threshold=float(p_threshold),
        z_window=int(z_window),
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        max_holding_days=int(max_holding_days),
        quality_filters={"enabled": use_filters, "min_r2": min_r2, "min_abs_beta": min_abs_beta, "min_half_life": min_half_life, "max_half_life": max_half_life},
    )
    if base_result is None:
        st.warning("Коинтегрированные пары не найдены.")
        return

    results_df = base_result["cointegration_analysis"]["results_df"]
    pair_options = results_df["pair"].tolist() if not results_df.empty else []
    selected_pair = None
    if selection_mode == "manual" and pair_options:
        selected_label = st.selectbox("Выберите пару", pair_options)
        left_t, right_t = selected_label.split("-")
        selected_pair = (left_t, right_t)

    result = run_full_pipeline(
        prices=prices,
        p_value_threshold=float(p_threshold),
        z_window=int(z_window),
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        max_holding_days=int(max_holding_days),
        selected_pair=selected_pair,
        selection_mode=selection_mode,
        quality_filters={"enabled": use_filters, "min_r2": min_r2, "min_abs_beta": min_abs_beta, "min_half_life": min_half_life, "max_half_life": max_half_life},
    )

    best_pair, metrics, corr_bt = result["best_pair"], result["metrics"], result.get("correlation_backtest")

    st.subheader("3.1 Описание экспериментальных данных")
    q = experimental_data["quality"]
    st.write(f"Период: **{q['Дата начала выборки']} — {q['Дата окончания выборки']}**, строк после очистки: **{q['Количество строк после очистки']}**.")
    st.dataframe(experimental_data["head"], use_container_width=True)
    st.dataframe(experimental_data["stats"], use_container_width=True)
    st.image(experimental_data["files"]["plot_png"], caption="Нормализованная динамика цен закрытия акций")

    st.subheader("3.2 Анализ коинтеграционных зависимостей")
    st.dataframe(results_df, use_container_width=True)
    st.markdown(f"Выбрана пара: **{best_pair['pair'][0]} - {best_pair['pair'][1]}** (p-value={best_pair['p_value']:.6f}, beta={best_pair['beta']:.4f}, R²={best_pair['r_squared']:.4f}, half-life={best_pair['half_life']:.1f}).")

    coint_heatmap_path = result["cointegration_analysis"].get("heatmap_path")
    corr_heatmap_path = result["comparison_section"].get("correlation_heatmap_path")
    if coint_heatmap_path:
        st.image(coint_heatmap_path, caption="Матрица p-value коинтеграции")
        st.caption("Светлые области соответствуют статистически значимым зависимостям между рядами (меньшие p-value).")
    if corr_heatmap_path:
        st.image(corr_heatmap_path, caption="Матрица коэффициентов корреляции")
        st.caption("Высокая корреляция сама по себе не гарантирует наличие коинтеграции и устойчивого спреда.")

    st.subheader("3.3 Анализ динамики спреда")
    spread_analysis = result["spread_analysis"]
    st.image(spread_analysis["files"]["spread_plot_png"], caption="Динамика спреда")
    st.image(spread_analysis["files"]["zscore_plot_png"], caption="Динамика Z-score")
    st.dataframe(spread_analysis["spread_df"][["spread", "z_score"]].dropna().tail(50), use_container_width=True)

    st.subheader("3.4 Реализация торговой стратегии")
    st.dataframe(result["trades"], use_container_width=True)

    st.subheader("3.5 Результаты бэктеста")
    c1, c2, c3 = st.columns(3)
    c1.metric("Совокупная доходность", f"{metrics['total_return']:.2%}")
    c2.metric("Коэффициент Шарпа", f"{metrics['sharpe_ratio']:.2f}")
    c3.metric("Макс. просадка", f"{metrics['max_drawdown']:.2%}")
    st.plotly_chart(render_equity_chart(result["equity"]), use_container_width=True)

    st.subheader("3.6 Сравнение методов: коинтеграция и корреляция")
    if corr_bt is None:
        st.info("Корреляционный бенчмарк недоступен.")
    else:
        st.info(build_method_comparison_text(metrics, corr_bt.get("metrics", {})))
        cmp = go.Figure()
        cmp.add_trace(go.Scatter(x=result["equity"].index, y=result["equity"].values, name="Коинтеграция"))
        cmp.add_trace(go.Scatter(x=corr_bt["equity"].index, y=corr_bt["equity"].values, name="Корреляция"))
        cmp.update_layout(title="Сравнение результатов при выбранных параметрах", xaxis_title="Дата", yaxis_title="Индекс капитала")
        st.plotly_chart(cmp, use_container_width=True)


if __name__ == "__main__":
    main()
