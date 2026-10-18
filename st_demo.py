from __future__ import annotations

import json
import time
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_echarts import st_echarts

from embedding import Embeddings
from main import (
    Node,
    Record,
    SearchResult,
    SemanticBTree,
    build_tree_from_records,
    run_benchmark,
    validate_record,
)


def inject_liquid_glass_theme() -> None:
    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap");

        :root {
            --glass-bg: rgba(255, 255, 255, 0.22);
            --glass-bg-strong: rgba(255, 255, 255, 0.34);
            --glass-border: rgba(255, 255, 255, 0.52);
            --glass-shadow: 0 28px 90px rgba(38, 86, 126, 0.26);
            --ink: #10243a;
            --muted: rgba(16, 36, 58, 0.68);
            --aqua: #9ee7f5;
            --coral: #ff815f;
            --rose: #f5a5d1;
            --gold: #ffd26a;
        }

        html, body, .stApp,
        button, input, textarea, select {
            font-family: "Manrope", "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif;
        }

        .stApp {
            color: var(--ink);
            background:
                linear-gradient(125deg, rgba(88, 151, 202, 0.96), rgba(202, 246, 237, 0.82) 26%, rgba(255, 222, 142, 0.66) 48%, rgba(248, 139, 129, 0.64) 68%, rgba(90, 166, 224, 0.95)),
                linear-gradient(210deg, rgba(255, 255, 255, 0.28), rgba(255, 255, 255, 0));
            overflow-x: hidden;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: -22%;
            pointer-events: none;
            background:
                conic-gradient(from 220deg at 48% 42%, rgba(255, 255, 255, 0.74), rgba(117, 223, 246, 0.48), rgba(255, 129, 95, 0.58), rgba(255, 210, 106, 0.44), rgba(255, 255, 255, 0.68)),
                linear-gradient(62deg, rgba(255, 255, 255, 0.36), transparent 34%, rgba(255, 255, 255, 0.22) 62%, transparent);
            filter: blur(42px) saturate(142%);
            opacity: 0.66;
            transform: rotate(-9deg) scale(1.08);
            z-index: 0;
        }

        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(120deg, rgba(255, 255, 255, 0.42), transparent 22%, rgba(255, 255, 255, 0.2) 42%, transparent 64%),
                repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0 1px, transparent 1px 14px);
            mix-blend-mode: soft-light;
            opacity: 0.46;
            z-index: 0;
        }

        .block-container,
        section[data-testid="stSidebar"] > div {
            position: relative;
            z-index: 1;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.45rem;
            padding-bottom: 2rem;
        }

        .liquid-hero {
            position: relative;
            isolation: isolate;
            min-height: 112px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            margin: 0.2rem 0 1.1rem;
            padding: 24px 32px;
            border: 1px solid rgba(255, 255, 255, 0.62);
            border-radius: 34px;
            background:
                linear-gradient(90deg, rgba(255, 255, 255, 0.22), rgba(255, 255, 255, 0.48) 40%, rgba(160, 232, 248, 0.42)),
                linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.34));
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.9),
                inset 0 -22px 46px rgba(114, 184, 224, 0.28),
                0 28px 80px rgba(36, 88, 127, 0.3);
            -webkit-backdrop-filter: blur(28px) saturate(150%);
            backdrop-filter: blur(28px) saturate(150%);
            overflow: hidden;
        }

        .liquid-hero::before {
            content: "";
            position: absolute;
            inset: 8px 18px auto 22px;
            height: 38%;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255, 255, 255, 0.62), transparent 72%);
            filter: blur(6px);
            opacity: 0.5;
            z-index: -1;
        }

        .liquid-hero::after {
            content: "";
            position: absolute;
            inset: auto 32px 14px 38%;
            height: 28%;
            border-radius: 999px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.46), transparent);
            filter: blur(10px);
            opacity: 0.62;
            z-index: -1;
        }

        .hero-copy {
            min-width: 0;
        }

        .liquid-hero h1 {
            display: block;
            margin: 0.12rem 0 0;
            color: #ffffff;
            font-size: 48px;
            font-weight: 700;
            line-height: 1;
            letter-spacing: 0;
            text-shadow: 0 12px 34px rgba(21, 55, 90, 0.34);
        }

        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 5px 12px;
            border: 1px solid rgba(255, 255, 255, 0.48);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.18);
            color: rgba(255, 255, 255, 0.88);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-subtitle {
            max-width: 680px;
            margin: 0.58rem 0 0;
            color: rgba(255, 255, 255, 0.82);
            font-size: 0.96rem;
            font-weight: 500;
            line-height: 1.55;
            text-shadow: 0 10px 26px rgba(21, 55, 90, 0.24);
        }

        .hero-badge {
            flex: 0 0 auto;
            min-width: 155px;
            padding: 18px 20px;
            border: 1px solid rgba(255, 255, 255, 0.58);
            border-radius: 26px;
            background: rgba(255, 255, 255, 0.18);
            color: #ffffff;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72), 0 18px 40px rgba(41, 85, 122, 0.18);
            text-align: right;
        }

        .hero-badge strong {
            display: block;
            font-size: 1.3rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: 0;
        }

        .hero-badge span {
            display: block;
            margin-top: 7px;
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.82rem;
            font-weight: 600;
        }

        h1 {
            display: none;
        }

        h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        h2 {
            margin-top: 0.35rem;
            font-size: 1.22rem;
            font-weight: 650;
        }

        p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        section[data-testid="stSidebar"] {
            background: rgba(238, 250, 255, 0.33);
            border-right: 1px solid rgba(255, 255, 255, 0.48);
            box-shadow: 16px 0 70px rgba(40, 91, 126, 0.18);
            -webkit-backdrop-filter: blur(26px) saturate(145%);
            backdrop-filter: blur(26px) saturate(145%);
        }

        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.48);
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.22);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.62);
            overflow: hidden;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            padding: 0 0.12rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 42px;
            border: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: 999px;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.74), rgba(255, 255, 255, 0.24)),
                linear-gradient(90deg, rgba(158, 231, 245, 0.45), rgba(255, 129, 95, 0.3));
            color: var(--ink);
            font-weight: 650;
            box-shadow: 0 12px 30px rgba(49, 101, 141, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.88);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(255, 255, 255, 0.95);
            color: var(--ink);
            transform: translateY(-1px);
            box-shadow: 0 18px 38px rgba(49, 101, 141, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.95);
        }

        .stTextInput input,
        .stTextArea textarea {
            border: 1px solid rgba(255, 255, 255, 0.58);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.34);
            color: var(--ink);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
            -webkit-backdrop-filter: blur(18px);
            backdrop-filter: blur(18px);
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: rgba(255, 255, 255, 0.9);
            box-shadow: 0 0 0 3px rgba(158, 231, 245, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.84);
        }

        div[data-baseweb="slider"] div[role="slider"] {
            background: #ffffff;
            box-shadow: 0 6px 20px rgba(57, 118, 155, 0.28);
        }

        div[data-baseweb="slider"] > div > div {
            background: linear-gradient(90deg, var(--aqua), var(--gold), var(--coral));
        }

        [data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.2);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.66);
            overflow: hidden;
        }

        [data-testid="stTable"] {
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.48);
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.2);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.62);
        }

        [data-testid="stTable"] table {
            background: transparent;
        }

        [data-testid="stTable"] th {
            color: var(--ink);
            background: rgba(255, 255, 255, 0.34);
        }

        [data-testid="stTable"] td {
            color: rgba(16, 36, 58, 0.82);
            background: rgba(255, 255, 255, 0.12);
        }

        .result-card,
        .empty-search-card {
            border: 1px solid rgba(255, 255, 255, 0.54);
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.36), rgba(255, 255, 255, 0.14)),
                rgba(255, 255, 255, 0.14);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72), 0 20px 48px rgba(37, 86, 122, 0.18);
            -webkit-backdrop-filter: blur(22px) saturate(145%);
            backdrop-filter: blur(22px) saturate(145%);
        }

        [data-testid="stMetric"] {
            min-height: 96px;
            padding: 17px 18px;
            border: 1px solid rgba(255, 255, 255, 0.54);
            border-radius: 24px;
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.42), rgba(255, 255, 255, 0.16)),
                rgba(255, 255, 255, 0.16);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72), 0 18px 42px rgba(37, 86, 122, 0.16);
            -webkit-backdrop-filter: blur(20px) saturate(145%);
            backdrop-filter: blur(20px) saturate(145%);
        }

        [data-testid="stMetric"] label {
            color: rgba(16, 36, 58, 0.6);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 800;
            letter-spacing: 0;
        }

        [data-testid="stMetricDelta"] {
            color: rgba(16, 36, 58, 0.58);
            font-weight: 700;
        }

        .result-card {
            position: relative;
            padding: 18px 18px 16px;
            margin: 0 0 13px;
            border-radius: 24px;
            overflow: hidden;
        }

        .result-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: linear-gradient(180deg, var(--aqua), var(--gold), var(--coral));
            opacity: 0.95;
        }

        .result-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
        }

        .result-rank {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.42);
            color: var(--ink);
            font-weight: 800;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82);
        }

        .result-id {
            color: rgba(16, 36, 58, 0.66);
            font-size: 0.86rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .result-text {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 600;
            line-height: 1.55;
        }

        .result-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 13px 0 11px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 5px 10px;
            border: 1px solid rgba(255, 255, 255, 0.54);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.24);
            color: rgba(16, 36, 58, 0.72);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .score-track {
            height: 8px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.34);
            overflow: hidden;
            box-shadow: inset 0 1px 2px rgba(37, 86, 122, 0.12);
        }

        .score-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--aqua), var(--gold), var(--coral));
        }

        .path-line {
            margin-top: 10px;
            color: rgba(16, 36, 58, 0.56);
            font-size: 0.78rem;
            font-weight: 650;
            line-height: 1.45;
            word-break: break-word;
        }

        .empty-search-card {
            padding: 22px;
            border-radius: 24px;
        }

        .empty-search-card strong {
            display: block;
            color: var(--ink);
            font-size: 1rem;
            margin-bottom: 7px;
        }

        .empty-search-card span {
            color: rgba(16, 36, 58, 0.62);
            font-weight: 600;
            line-height: 1.55;
        }

        .stAlert {
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.48);
            background: rgba(255, 255, 255, 0.24);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.62);
        }

        .element-container iframe {
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.14);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.62), 0 22px 54px rgba(37, 86, 122, 0.18);
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .liquid-hero {
                min-height: 104px;
                padding: 20px 26px;
                border-radius: 34px;
            }

            .liquid-hero h1 {
                font-size: 38px;
            }

            .hero-subtitle,
            .hero-badge {
                display: none;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                padding: 0;
            }

        }

        @media (max-width: 1100px) and (min-width: 761px) {
            .liquid-hero h1 {
                font-size: 40px;
            }

            .hero-badge {
                display: none;
            }
        }
        </style>
        <div class="liquid-hero">
            <div class="hero-copy">
                <span class="hero-eyebrow">Semantic index lab</span>
                <h1>Semantic Search</h1>
                <p class="hero-subtitle">
                    Поиск ближайших текстов по эмбеддингам, визуализация дерева и
                    сравнение с линейным перебором в одном спокойном glass-интерфейсе.
                </p>
            </div>
            <div class="hero-badge">
                <strong>O(k log N)</strong>
                <span>дерево + путь до листа</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_embedder() -> Embeddings:
    return Embeddings()


def load_records(file_path: Path) -> list[Record]:
    with file_path.open("r", encoding="utf-8") as file:
        raw: object = json.load(file)

    if not isinstance(raw, list):
        raise ValueError("initial_data.json должен содержать список записей")

    return [validate_record(entry) for entry in raw]


def initialize_records(file_path: Path) -> None:
    if "records" not in st.session_state:
        st.session_state.records = load_records(file_path)


def invalidate_tree() -> None:
    st.session_state.pop("tree", None)
    st.session_state.pop("tree_key", None)
    st.session_state.pop("last_search_path", None)


def get_next_record_id() -> int:
    records: list[Record] = st.session_state.records
    if not records:
        return 1
    return max(record.id for record in records) + 1


SAMPLE_QUERIES = [
    "космический корабль и спутники",
    "нейронные сети и обработка данных",
    "врачебные исследования и анализ крови",
    "музыкант на сцене перед концертом",
    "спортсмены и решающий матч",
    "историческая находка археологов",
]


def format_metric_value(value: int | float | str | None) -> str:
    if value is None:
        return "-"

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"

    return str(value)


def render_metric_grid(items: list[tuple[str, int | float | str | None, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items, strict=True):
        column.metric(label=label, value=format_metric_value(value), help=note)


def collect_tree_stats(root: Node) -> dict[str, int | float]:
    stack = [root]
    node_count = 0
    leaf_count = 0
    item_count = 0
    max_depth = 0
    radius_sum = 0.0

    while stack:
        node = stack.pop()
        node_count += 1
        max_depth = max(max_depth, node.depth)
        radius_sum += node.radius

        if node.is_leaf:
            leaf_count += 1
            item_count += len(node.items)
        else:
            stack.extend(node.children)

    return {
        "nodes": node_count,
        "leaves": leaf_count,
        "items": item_count,
        "height": max_depth + 1,
        "avg_leaf_size": item_count / leaf_count if leaf_count else 0.0,
        "avg_radius": radius_sum / node_count if node_count else 0.0,
    }


def render_index_overview(
    records: list[Record],
    tree: SemanticBTree | None,
    leaf_size: int,
    k_value: int,
) -> None:
    if tree is None:
        render_metric_grid(
            [
                ("Записей", len(records), "корпус сейчас пуст"),
                ("Лист", leaf_size, "настройка дерева"),
                ("Top-K", k_value, "сколько результатов показывать"),
                ("Размерность", "-", "появится после построения"),
            ],
        )
        return

    stats = collect_tree_stats(tree.root)
    render_metric_grid(
        [
            ("Записей", len(records), "индексированный корпус"),
            ("Узлов", int(stats["nodes"]), f"листьев: {int(stats['leaves'])}"),
            ("Высота", int(stats["height"]), f"лист до {leaf_size} элементов"),
            ("Размерность", tree.dim, f"Top-K: {k_value}"),
        ],
    )


def render_query_presets() -> None:
    if "query_text" not in st.session_state:
        st.session_state.query_text = SAMPLE_QUERIES[0]

    selected_query = st.sidebar.selectbox(
        "Быстрый пример запроса",
        SAMPLE_QUERIES,
        index=0,
    )

    if st.sidebar.button("Подставить пример", use_container_width=True):
        st.session_state.query_text = selected_query


def render_empty_search_state() -> None:
    st.info(
        "Введите запрос и нажмите «Искать»: приложение покажет ближайшие тексты, "
        "расстояния, score и путь по дереву до лучшего листа.",
    )


def render_search_summary(
    results: list[SearchResult],
    embedding_ms: float,
    search_ms: float,
) -> None:
    best = results[0] if results else None
    path_length = len(best.path) if best is not None else 0
    render_metric_grid(
        [
            ("Найдено", len(results), "результатов в выдаче"),
            ("Лучший score", best.score if best is not None else None, "чем выше, тем ближе"),
            ("Эмбеддинг", f"{embedding_ms:.1f} ms", "вектор запроса"),
            ("Поиск", f"{search_ms:.1f} ms", f"глубина пути: {path_length}"),
        ],
    )


def render_result_cards(results: list[SearchResult]) -> None:
    if not results:
        st.info("Ничего не найдено")
        return

    for place, result in enumerate(results, start=1):
        text = result.payload.text if result.payload is not None else ""
        score_value = max(0.0, min(1.0, result.score))
        path = " -> ".join(result.path)

        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            top_left.markdown(f"**{place}. id {escape(result.id)}**")
            top_right.markdown(f"**score {result.score:.4f}**")
            st.write(text)
            st.progress(
                score_value,
                text=f"distance {result.distance:.4f}",
            )
            st.caption(f"path: {path}")


def add_record(text: str) -> tuple[bool, str]:
    normalized_text = text.strip()
    if not normalized_text:
        return False, "Введите текст записи"

    new_id = get_next_record_id()
    records: list[Record] = st.session_state.records
    st.session_state.records = [*records, Record(id=new_id, text=normalized_text)]
    invalidate_tree()
    return True, f"Запись с id={new_id} добавлена"


def delete_record(record_id: str) -> tuple[bool, str]:
    normalized_id = record_id.strip()
    if not normalized_id:
        return False, "Введите id для удаления"

    records: list[Record] = st.session_state.records
    filtered = [record for record in records if str(record.id) != normalized_id]

    if len(filtered) == len(records):
        return False, f"Запись с id={normalized_id} не найдена"

    st.session_state.records = filtered
    invalidate_tree()
    return True, f"Запись с id={normalized_id} удалена"


def get_or_build_tree(
    records: list[Record],
    embedder: Embeddings,
    leaf_size: int,
) -> SemanticBTree | None:
    if not records:
        return None

    signature = tuple((record.id, record.text) for record in records)
    current_key = (leaf_size, signature)

    if st.session_state.get("tree_key") != current_key:
        st.session_state.tree = build_tree_from_records(
            records=records,
            embeddings=embedder,
            t=leaf_size,
        )
        st.session_state.tree_key = current_key

    return st.session_state.tree  # type: ignore[no-any-return]


def make_node_label(node: Node, is_root: bool) -> str:
    kind = "leaf" if node.is_leaf else "split"
    if is_root:
        max_id = node.max_item_id if node.max_item_id is not None else "-"
        return (
            f"{node.node_id}\n"
            f"{kind}\n"
            f"size={node.size}\n"
            f"depth={node.depth}\n"
            f"radius={node.radius:.3f}\n"
            f"max_id={max_id}"
        )

    return (
        f"{node.node_id}\n"
        f"{kind}\n"
        f"size={node.size}\n"
        f"depth={node.depth}\n"
        f"radius={node.radius:.3f}"
    )


def make_leaf_label(item_id: str, payload: Record | None) -> str:
    if payload is None:
        return f"id={item_id}"

    text = payload.text
    if len(text) > 75:
        text = text[:72] + "..."

    return f"id={payload.id}\n{text}"


def tree_to_echarts(
    node: Node,
    highlight_path: set[str],
    depth: int = 0,
    is_root: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": make_node_label(node=node, is_root=is_root),
        "collapsed": depth >= 3 and node.node_id not in highlight_path,
    }

    if node.node_id in highlight_path:
        data["lineStyle"] = {"width": 4, "color": "#ff815f"}
        data["itemStyle"] = {
            "borderWidth": 4,
            "borderColor": "#ff815f",
            "color": "rgba(255, 255, 255, 0.96)",
        }

    if node.is_leaf:
        data["children"] = [
            {
                "name": make_leaf_label(item_id=item.item_id, payload=item.payload),
                "value": item.item_id,
            }
            for item in node.items
        ]
        return data

    data["children"] = [
        tree_to_echarts(
            node=child,
            highlight_path=highlight_path,
            depth=depth + 1,
        )
        for child in node.children
    ]
    return data


def render_tree(tree: SemanticBTree, search_path: tuple[str, ...]) -> None:
    data = tree_to_echarts(
        node=tree.root,
        highlight_path=set(search_path),
        is_root=True,
    )

    options = {
        "backgroundColor": "transparent",
        "color": ["#ffffff", "#9ee7f5", "#ffd26a", "#ff815f", "#f5a5d1"],
        "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
        "series": [
            {
                "type": "tree",
                "data": [data],
                "top": "2%",
                "left": "10%",
                "bottom": "2%",
                "right": "30%",
                "symbol": "circle",
                "symbolSize": 10,
                "orient": "LR",
                "layout": "orthogonal",
                "expandAndCollapse": True,
                "initialTreeDepth": -1,
                "edgeShape": "polyline",
                "edgeForkPosition": "50%",
                "roam": True,
                "lineStyle": {
                    "color": "rgba(255, 255, 255, 0.72)",
                    "width": 2,
                    "shadowColor": "rgba(56, 111, 152, 0.22)",
                    "shadowBlur": 8,
                },
                "itemStyle": {
                    "color": "rgba(255, 255, 255, 0.9)",
                    "borderColor": "rgba(158, 231, 245, 0.95)",
                    "borderWidth": 2,
                    "shadowColor": "rgba(56, 111, 152, 0.24)",
                    "shadowBlur": 14,
                },
                "label": {
                    "position": "left",
                    "verticalAlign": "middle",
                    "align": "right",
                    "fontSize": 12,
                    "lineHeight": 18,
                    "color": "#10243a",
                    "backgroundColor": "rgba(255, 255, 255, 0.46)",
                    "borderColor": "rgba(255, 255, 255, 0.72)",
                    "borderWidth": 1,
                    "borderRadius": 12,
                    "padding": [7, 10],
                    "shadowColor": "rgba(56, 111, 152, 0.18)",
                    "shadowBlur": 10,
                },
                "leaves": {
                    "itemStyle": {
                        "color": "rgba(255, 255, 255, 0.92)",
                        "borderColor": "rgba(255, 210, 106, 0.95)",
                    },
                    "label": {
                        "position": "right",
                        "verticalAlign": "middle",
                        "align": "left",
                        "color": "#10243a",
                        "lineHeight": 18,
                        "backgroundColor": "rgba(255, 255, 255, 0.42)",
                        "borderColor": "rgba(255, 255, 255, 0.7)",
                        "borderWidth": 1,
                        "borderRadius": 12,
                        "padding": [7, 10],
                        "shadowColor": "rgba(56, 111, 152, 0.16)",
                        "shadowBlur": 10,
                    },
                },
            },
        ],
    }

    st_echarts(options=options, height="850px", events={}, key="tree_chart")


def render_record_controls() -> None:
    with st.sidebar.expander("Управление данными", expanded=True):
        with st.form("add_record_form", clear_on_submit=True):
            text = st.text_area("Новый текст", height=110)
            if st.form_submit_button("Добавить"):
                success, message = add_record(text)
                st.session_state.feedback = ("success" if success else "error", message)
                if success:
                    st.rerun()

        with st.form("delete_record_form", clear_on_submit=True):
            record_id = st.text_input("id для удаления")
            if st.form_submit_button("Удалить"):
                success, message = delete_record(record_id)
                st.session_state.feedback = ("success" if success else "error", message)
                if success:
                    st.rerun()


def render_feedback() -> None:
    feedback: tuple[str, str] | None = st.session_state.pop("feedback", None)
    if feedback is None:
        return

    level, message = feedback
    if level == "success":
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)


def render_records_overview(records: list[Record]) -> None:
    st.sidebar.caption(f"Всего записей: {len(records)}")
    st.sidebar.caption(f"Следующий id: {get_next_record_id()}")

    ids = ", ".join(str(record.id) for record in records[:15])
    if ids:
        st.sidebar.caption(f"Первые id: {ids}")

    with st.sidebar.expander("Фрагмент корпуса"):
        preview_rows = [
            {"id": record.id, "text": record.text}
            for record in records[:8]
        ]
        st.dataframe(preview_rows, hide_index=True, use_container_width=True)


def render_search_results(
    tree: SemanticBTree | None,
    embedder: Embeddings,
    query: str,
    k_value: int,
) -> None:
    st.subheader("Поиск")

    if tree is None:
        st.info("Данных пока нет")
        return

    if not st.button("Искать", use_container_width=True):
        render_empty_search_state()
        return

    normalized_query = query.strip()
    if not normalized_query:
        st.warning("Введите запрос")
        return

    embedding_started = time.perf_counter()
    query_vector = embedder.embed(normalized_query)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000

    search_started = time.perf_counter()
    results: list[SearchResult] = tree.search(query=query_vector, k=k_value)
    search_ms = (time.perf_counter() - search_started) * 1000

    if results:
        st.session_state.last_search_path = results[0].path

    render_search_summary(
        results=results,
        embedding_ms=embedding_ms,
        search_ms=search_ms,
    )
    render_result_cards(results)

    rows = [
        {
            "id": result.id,
            "distance": round(result.distance, 4),
            "score": round(result.score, 4),
            "path": " -> ".join(result.path),
            "text": result.payload.text if result.payload is not None else "",
        }
        for result in results
    ]

    with st.expander("Таблица результатов"):
        st.table(rows)


def render_tree_panel(tree: SemanticBTree | None) -> None:
    st.subheader("Структура дерева")

    if tree is None:
        st.info("Дерево пустое")
        return

    search_path: tuple[str, ...] = st.session_state.get("last_search_path", tuple())
    if search_path:
        st.caption("Подсвечен путь до листа лучшего результата: " + " -> ".join(search_path))

    render_tree(tree=tree, search_path=search_path)


def render_benchmark_panel() -> None:
    with st.expander("Бенчмарк линейного поиска и дерева"):
        st.write(
            "Замер делается на синтетических нормированных векторах размерности 384, "
            "чтобы отдельно проверить саму структуру данных без ожидания модели эмбеддингов."
        )

        if not st.button("Запустить бенчмарк", use_container_width=True):
            return

        rows = run_benchmark(repeats=3)
        table = [
            {
                "N": row.n,
                "Линейный, ms": round(row.linear_ms, 3),
                "Дерево, ms": round(row.tree_ms, 3),
                "Ускорение": f"{row.speedup:.2f}x",
            }
            for row in rows
        ]
        st.table(table)

        options = {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {
                "data": ["Линейный", "Дерево"],
                "textStyle": {"color": "#10243a"},
            },
            "grid": {"left": 46, "right": 18, "top": 48, "bottom": 34},
            "xAxis": {
                "type": "category",
                "data": [str(row.n) for row in rows],
                "axisLine": {"lineStyle": {"color": "rgba(16, 36, 58, 0.35)"}},
                "axisLabel": {"color": "rgba(16, 36, 58, 0.74)"},
            },
            "yAxis": {
                "type": "value",
                "name": "ms",
                "nameTextStyle": {"color": "rgba(16, 36, 58, 0.7)"},
                "splitLine": {"lineStyle": {"color": "rgba(255, 255, 255, 0.38)"}},
                "axisLabel": {"color": "rgba(16, 36, 58, 0.74)"},
            },
            "series": [
                {
                    "name": "Линейный",
                    "type": "line",
                    "smooth": True,
                    "symbolSize": 8,
                    "lineStyle": {"width": 4, "color": "#ff815f"},
                    "itemStyle": {"color": "#ff815f"},
                    "areaStyle": {"color": "rgba(255, 129, 95, 0.12)"},
                    "data": [round(row.linear_ms, 3) for row in rows],
                },
                {
                    "name": "Дерево",
                    "type": "line",
                    "smooth": True,
                    "symbolSize": 8,
                    "lineStyle": {"width": 4, "color": "#3bc4df"},
                    "itemStyle": {"color": "#3bc4df"},
                    "areaStyle": {"color": "rgba(59, 196, 223, 0.16)"},
                    "data": [round(row.tree_ms, 3) for row in rows],
                },
            ],
        }
        st_echarts(options=options, height="360px", events={}, key="benchmark_chart")


def main() -> None:
    st.set_page_config(page_title="Semantic Search", layout="wide")
    inject_liquid_glass_theme()
    st.title("Семантический поиск по эмбеддингам")

    data_path = Path(__file__).with_name("initial_data.json")
    initialize_records(data_path)
    embedder = get_embedder()

    leaf_size = st.sidebar.slider("Размер листа", 2, 40, 8)
    k_value = st.sidebar.slider("Top-K", 1, 20, 5)
    render_record_controls()
    render_feedback()

    records: list[Record] = st.session_state.records
    render_records_overview(records)

    tree = get_or_build_tree(records=records, embedder=embedder, leaf_size=leaf_size)
    render_index_overview(
        records=records,
        tree=tree,
        leaf_size=leaf_size,
        k_value=k_value,
    )
    render_query_presets()

    query = st.text_input("Введите запрос", key="query_text")
    left, right = st.columns([1, 2])

    with left:
        render_search_results(tree=tree, embedder=embedder, query=query, k_value=k_value)
        render_benchmark_panel()

    with right:
        render_tree_panel(tree=tree)


if __name__ == "__main__":
    main()
