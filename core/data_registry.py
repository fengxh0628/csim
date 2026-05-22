"""Central data registry with DAG-based dependency resolution.

Data managers register dependencies (fields they need from other dmgrs).
The registry builds a DAG and determines execution order via topological sort.
"""
import numpy as np
from collections import defaultdict, deque


class DataRegistry:

    def __init__(self):
        self.data_dict: dict[str, np.ndarray] = {}
        self._data_dmgr_mapping: dict[str, str] = {}  # field -> dmgr_id
        self._dependencies: dict[str, list[str]] = defaultdict(list)  # dmgr_id -> [field names it depends on]

    def getdata(self, field: str) -> np.ndarray:
        if field not in self.data_dict:
            raise KeyError(f"Data field '{field}' not found in registry")
        return self.data_dict[field]

    def setdata(self, field: str, data: np.ndarray):
        self.data_dict[field] = data

    def has(self, field: str) -> bool:
        return field in self.data_dict

    def set_dmgr_from_data(self, field: str, dmgr_id: str):
        """Record which dmgr produces this field."""
        self._data_dmgr_mapping[field] = dmgr_id

    def register_dependency(self, dmgr_id: str, field: str):
        """Record that dmgr_id depends on field (produced by another dmgr)."""
        self._dependencies[dmgr_id].append(field)

    def get_execution_order(self, dmgr_ids: list[str]) -> list[str]:
        """Topological sort of data managers based on dependencies."""
        # Build adjacency: edge from A -> B means A must run before B
        graph = defaultdict(set)  # node -> set of successors
        in_degree = {mid: 0 for mid in dmgr_ids}
        dmgr_set = set(dmgr_ids)

        for mid in dmgr_ids:
            for field in self._dependencies.get(mid, []):
                dep_dmgr = self._data_dmgr_mapping.get(field)
                if dep_dmgr and dep_dmgr in dmgr_set and dep_dmgr != mid:
                    graph[dep_dmgr].add(mid)
                    in_degree[mid] = in_degree.get(mid, 0) + 1

        # Kahn's algorithm
        queue = deque([mid for mid in dmgr_ids if in_degree[mid] == 0])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for succ in graph[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(dmgr_ids):
            remaining = set(dmgr_ids) - set(result)
            raise ValueError(f"Circular dependency detected: {remaining}")

        return result


_registry: DataRegistry = None


def get_data_registry() -> DataRegistry:
    global _registry
    if _registry is None:
        _registry = DataRegistry()
    return _registry
