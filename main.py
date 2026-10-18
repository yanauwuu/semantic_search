from __future__ import annotations
import heapq
import math
import time
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Record:
    id: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    id: str
    distance: float
    score: float
    path: tuple[str, ...]
    payload: Record | None


@dataclass(frozen=True)
class TreeItem:
    vector: np.ndarray
    item_id: str
    payload: Record | None


@dataclass
class Node:
    node_id: str
    center: np.ndarray
    radius: float
    size: int
    depth: int
    items: list[TreeItem]
    children: list[Node]

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def max_item_id(self) -> int | None:
        ids: list[int] = []

        for item in self.items:
            if item.payload is not None:
                ids.append(item.payload.id)

        for child in self.children:
            child_max_id = child.max_item_id
            if child_max_id is not None:
                ids.append(child_max_id)

        if not ids:
            return None
        return max(ids)


@dataclass(frozen=True)
class BenchmarkRow:
    n: int
    linear_ms: float
    tree_ms: float
    speedup: float


class EmbeddingModel(Protocol):
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        ...


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32)

    if result.ndim != 1:
        raise ValueError("vector должен быть одномерным")

    norm = float(np.linalg.norm(result))
    if norm == 0.0:
        raise ValueError("нельзя добавить нулевой вектор")

    return result / norm


def euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def make_score(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def validate_record(entry: object) -> Record:
    if not isinstance(entry, dict):
        raise ValueError("каждая запись должна быть объектом")

    raw_id = entry.get("id")
    raw_text = entry.get("text")

    try:
        record_id = int(raw_id)
    except (TypeError, ValueError) as error:
        raise ValueError("у записи должен быть целочисленный id") from error

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError(f"у записи id={record_id} должен быть непустой text")

    return Record(id=record_id, text=raw_text.strip())


class SemanticBTree:
    def __init__(self, leaf_size: int = 8, t: int | None = None) -> None:
        self.leaf_size = max(1, int(t if t is not None else leaf_size))
        self._items: list[TreeItem] = []
        self._root: Node | None = None
        self._need_rebuild = True
        self.dim: int | None = None

    @property
    def root(self) -> Node:
        self.rebuild()
        if self._root is None:
            raise ValueError("индекс пустой")

        return self._root

    @property
    def size(self) -> int:
        return len(self._items)

    def insert(self, vector: np.ndarray, item_id: str, payload: Record | None = None) -> None:
        normalized = normalize_vector(vector)
        if self.dim is None:
            self.dim = int(normalized.shape[0])
        elif normalized.shape[0] != self.dim:
            raise ValueError("размерность вектора не совпадает с индексом")

        self._items.append(
            TreeItem(vector=normalized, item_id=str(item_id), payload=payload),
        )
        self._need_rebuild = True

    def rebuild(self) -> None:
        if not self._need_rebuild:
            return

        if not self._items:
            self._root = None
            self._need_rebuild = False
            return

        self._root = self._build_node(items=list(self._items), node_id="root", depth=0)
        self._need_rebuild = False

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        if k <= 0 or not self._items:
            return []

        self.rebuild()
        if self._root is None:
            return []

        normalized_query = normalize_vector(query)
        if self.dim is not None and normalized_query.shape[0] != self.dim:
            raise ValueError("размерность запроса не совпадает с индексом")

        limit = min(k, len(self._items))
        best: list[tuple[float, int, TreeItem, tuple[str, ...]]] = []
        counter = 0

        def add_candidate(item: TreeItem, path: tuple[str, ...]) -> None:
            nonlocal counter
            distance = euclidean_distance(normalized_query, item.vector)
            heap_entry = (-distance, counter, item, path)
            counter += 1

            if len(best) < limit:
                heapq.heappush(best, heap_entry)
                return

            if distance < -best[0][0]:
                heapq.heapreplace(best, heap_entry)

        def visit(node: Node, path: tuple[str, ...]) -> None:
            distance_to_ball = euclidean_distance(normalized_query, node.center)
            lower_bound = max(0.0, distance_to_ball - node.radius)

            if len(best) >= limit and lower_bound > -best[0][0]:
                return

            if node.is_leaf:
                for item in node.items:
                    add_candidate(item=item, path=path)
                return

            ordered_children = sorted(
                node.children,
                key=lambda child: max(0.0,
                    euclidean_distance(normalized_query, child.center) - child.radius,
                ),
            )

            for child in ordered_children:
                visit(child, (*path, child.node_id))

        visit(self._root, (self._root.node_id,))
        sorted_items = sorted(best, key=lambda entry: -entry[0])
        results: list[SearchResult] = []

        for negative_distance, _, item, path in sorted_items:
            distance = -negative_distance
            results.append(
                SearchResult(id=item.item_id, distance=distance,
                    score=make_score(distance), path=path,
                    payload=item.payload),
            )

        return results

    def _build_node(self, items: list[TreeItem], node_id: str, depth: int) -> Node:
        vectors = np.stack([item.vector for item in items])
        center = np.mean(vectors, axis=0).astype(np.float32)
        radius = max(euclidean_distance(vector, center) for vector in vectors)

        if len(items) <= self.leaf_size:
            return Node(node_id=node_id, center=center,
                radius=radius, size=len(items), depth=depth,
                items=items, children=[])
        left_items, right_items = self._split_items(items=items, vectors=vectors)

        return Node(
            node_id=node_id,
            center=center,
            radius=radius,
            size=len(items),
            depth=depth,
            items=[],
            children=[
                self._build_node(left_items, f"{node_id}.L", depth + 1),
                self._build_node(right_items, f"{node_id}.R", depth + 1),
            ],
        )

    def _split_items(self, items: list[TreeItem], vectors: np.ndarray) -> tuple[list[TreeItem], list[TreeItem]]:
        center = np.mean(vectors, axis=0)
        first_index = int(np.argmax(np.linalg.norm(vectors - center, axis=1)))
        first_vector = vectors[first_index]
        second_index = int(np.argmax(np.linalg.norm(vectors - first_vector, axis=1)))
        second_vector = vectors[second_index]
        left_items: list[TreeItem] = []
        right_items: list[TreeItem] = []

        for item in items:
            dist_left = euclidean_distance(item.vector, first_vector)
            dist_right = euclidean_distance(item.vector, second_vector)
            if dist_left <= dist_right:
                left_items.append(item)
            else:
                right_items.append(item)

        if left_items and right_items:
            return left_items, right_items

        # если точки почти одинаковые, делим по координате с максимальным разбросом
        axis = int(np.argmax(np.var(vectors, axis=0)))
        ordered = sorted(items, key=lambda item: float(item.vector[axis]))
        middle = len(ordered) // 2
        return ordered[:middle], ordered[middle:]


def build_tree_from_records(records: list[Record], embeddings: EmbeddingModel, t: int = 8,) -> SemanticBTree:
    tree = SemanticBTree(leaf_size=t)
    if not records:
        return tree

    texts = [record.text for record in records]
    vectors = embeddings.embed_batch(texts)
    for record, vector in zip(records, vectors, strict=True):
        tree.insert(vector=vector, item_id=str(record.id), payload=record)

    tree.rebuild()
    return tree


def linear_search(vectors: np.ndarray, payloads: list[Record],
    query: np.ndarray, k: int) -> list[SearchResult]:
    if k <= 0 or len(payloads) == 0:
        return []

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors должен быть матрицей размера N x dim")

    if matrix.shape[0] != len(payloads):
        raise ValueError("число векторов и payload должно совпадать")

    normalized_query = normalize_vector(query)
    if matrix.shape[1] != normalized_query.shape[0]:
        raise ValueError("размерность запроса не совпадает с vectors")

    limit = min(k, len(payloads))
    best: list[tuple[float, int, Record]] = []

    for index, record in enumerate(payloads):
        vector = normalize_vector(matrix[index])
        distance = euclidean_distance(normalized_query, vector)
        heap_entry = (-distance, index, record)

        if len(best) < limit:
            heapq.heappush(best, heap_entry)
        elif distance < -best[0][0]:
            heapq.heapreplace(best, heap_entry)

    ordered = sorted(best, key=lambda entry: -entry[0])
    results: list[SearchResult] = []

    for negative_distance, _, record in ordered:
        distance = -negative_distance
        results.append(
            SearchResult(
                id=str(record.id),
                distance=distance,
                score=make_score(distance),
                path=("linear",),
                payload=record,
            ),
        )

    return results

def make_synthetic_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_count = max(4, min(32, n // 100))
    centers = rng.normal(size=(cluster_count, dim)).astype(np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    vectors = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        center = centers[i % cluster_count]
        noise = rng.normal(scale=0.08, size=dim).astype(np.float32)
        vector = center + noise
        vectors[i] = vector / np.linalg.norm(vector)

    return vectors


def make_synthetic_records(n: int) -> list[Record]:
    return [Record(id=i + 1, text=f"synthetic text {i + 1}") for i in range(n)]


def time_call(repeats: int, action: Callable[[], object]) -> float:
    timings: list[float] = []

    for _ in range(repeats):
        started = time.perf_counter()
        action()
        timings.append((time.perf_counter() - started) * 1000)

    return min(timings)


def run_benchmark(
    sizes: Sequence[int] = (100, 500, 1000, 5000, 10000), dim: int = 384,
    k: int = 10,repeats: int = 5) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    for n in sizes:
        vectors = make_synthetic_vectors(n=n, dim=dim, seed=100 + n)
        records = make_synthetic_records(n=n)
        query = vectors[0] + np.random.default_rng(7).normal(scale=0.03, size=dim)
        tree = SemanticBTree(leaf_size=16)
        for record, vector in zip(records, vectors, strict=True):
            tree.insert(vector=vector, item_id=str(record.id), payload=record)
        tree.rebuild()
        linear_ms = time_call(repeats=repeats,
            action=lambda: linear_search(vectors=vectors, payloads=records, query=query, k=k))
        tree_ms = time_call(repeats=repeats, action=lambda: tree.search(query=query, k=k))
        speedup = linear_ms / tree_ms if tree_ms > 0 else math.inf
        rows.append(BenchmarkRow(n=n, linear_ms=linear_ms, tree_ms=tree_ms, speedup=speedup))

    return rows


def format_benchmark_table(rows: list[BenchmarkRow]) -> str:
    header = "N       Линейный (ms)   Дерево (ms)   Ускорение"
    line = "------  -------------   -----------   ---------"
    table_rows = [header, line]

    for row in rows:
        table_rows.append(
            f"{row.n:<6}  {row.linear_ms:>13.3f}   "
            f"{row.tree_ms:>11.3f}   {row.speedup:>8.2f}x",
        )

    return "\n".join(table_rows)


def main() -> None:
    rows = run_benchmark()
    print(format_benchmark_table(rows))


if __name__ == "__main__":
    main()
