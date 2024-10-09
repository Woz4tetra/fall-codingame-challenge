from __future__ import annotations
import time

import zlib 
import json 
import sys 
import scipy.spatial  # type: ignore
from abc import ABC, abstractmethod 
from dataclasses import asdict, dataclass, field 
from enum import Enum 
from typing import Any, Callable, Generator 
import numpy as np 

from scipy.sparse.csgraph import shortest_path  # type: ignore

np.set_printoptions(threshold=sys.maxsize)


def debug_print(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _asdict_factory(data: Any) -> dict:
    def convert_value(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        return obj

    return dict((k, convert_value(v)) for k, v in data)


def to_dict(obj: Any) -> dict:
    return asdict(obj, dict_factory=_asdict_factory)


BuildingId = int
BuildingType = int

LANDING_PAD_BUILDING_TYPE = 0
POD_COST = 1000


class IndexPair:
    def __init__(self, index1: int, index2: int) -> None:
        self.index1 = index1
        self.index2 = index2

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return self.index1
        elif index == 1:
            return self.index2
        raise IndexError(f"Index out of bounds: {index}")

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Generator[int, None, None]:
        yield self.index1
        yield self.index2

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, IndexPair):
            return False
        return (self.index1 == other.index1 and self.index2 == other.index2) or (
            self.index1 == other.index2 and self.index2 == other.index1
        )

    def __hash__(self) -> int:
        return hash((self.index1, self.index2)) + hash((self.index2, self.index1))

    def __str__(self) -> str:
        return f"({self.index1}, {self.index2})"

    __repr__ = __str__


@dataclass
class TransportLine:
    building_id_1: BuildingId
    building_id_2: BuildingId
    capacity: int

    def to_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TransportLine:
        return cls(
            int(data["building_id_1"]),
            int(data["building_id_2"]),
            int(data["capacity"]),
        )


@dataclass
class Pod:
    pod_id: int
    itinerary: list[BuildingId]
    capacity: int = 10

    def to_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Pod:
        return cls(
            int(data["pod_id"]),
            [int(item) for item in data["itinerary"]],
            int(data["capacity"]),
        )


@dataclass
class Module:
    type: BuildingType
    id: BuildingId
    coordinates: tuple[int, int]

    def to_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Module:
        return cls(
            int(data["type"]),
            int(data["id"]),
            (int(data["coordinates"][0]), int(data["coordinates"][1])),
        )


@dataclass
class Astronaut:
    type: BuildingType

    def to_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Astronaut:
        return cls(int(data["type"]))


@dataclass
class LandingPad(Module):
    astronauts: list[Astronaut]

    def __post_init__(self) -> None:
        self.astronauts_type_map: dict[BuildingType, list[Astronaut]] = {
            astronaut.type: [astronaut] for astronaut in self.astronauts
        }

    def to_dict(self) -> dict:
        serialized = to_dict(self)
        serialized["astronauts"] = {
            type: len(astronauts)
            for type, astronauts in self.astronauts_type_map.items()
        }
        return serialized

    @classmethod
    def from_dict(cls, data: dict) -> LandingPad:
        astronaut_data = data["astronauts"]
        astronauts = []
        for type, count in astronaut_data.items():
            astronauts.extend([Astronaut(int(type)) for _ in range(count)])
        return cls(
            int(data["type"]),
            int(data["id"]),
            (int(data["coordinates"][0]), int(data["coordinates"][1])),
            astronauts,
        )


class InputSource(ABC):
    @abstractmethod
    def next_line(self) -> str: ...


class LiveInputSource(InputSource):
    def next_line(self) -> str:
        return input()


class BufferedInputSource(InputSource):
    def __init__(self) -> None:
        self.buffer: list[str] = []

    def append(self, line: str) -> str:
        self.buffer.append(line)
        return line

    def next_line(self) -> str:
        return self.buffer.pop(0)


@dataclass
class LunarMonthData:
    resources: int = 0
    transport_lines: list[TransportLine] = field(default_factory=list)
    pods: list[Pod] = field(default_factory=list)
    buildings: list[Module] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.buildings_by_type: dict[BuildingType, list[Module]] = {}
        for building in self.buildings:
            self.buildings_by_type.setdefault(building.type, []).append(building)

    def to_dict(self) -> dict:
        return {
            "resources": self.resources,
            "transport_lines": [t.to_dict() for t in self.transport_lines],
            "pods": [p.to_dict() for p in self.pods],
            "buildings": [b.to_dict() for b in self.buildings],
        }

    def to_compressed_string(self) -> bytes:
        return zlib.compress(json.dumps(self.to_dict()).encode("utf-8"))

    @classmethod
    def from_compressed_string(self, raw_data: bytes) -> LunarMonthData:
        return self.from_dict(json.loads(zlib.decompress(raw_data).decode("utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> LunarMonthData:
        return cls(
            int(data["resources"]),
            [TransportLine.from_dict(t) for t in data["transport_lines"]],
            [Pod.from_dict(p) for p in data["pods"]],
            [
                LandingPad.from_dict(b)
                if b["type"] == LANDING_PAD_BUILDING_TYPE
                else Module.from_dict(b)
                for b in data["buildings"]
            ],
        )

    def update_from_input(self, input_source: InputSource) -> int:
        self.resources = int(input_source.next_line())
        num_travel_routes = int(input_source.next_line())
        transport_lines = []
        for i in range(num_travel_routes):
            building_id_1, building_id_2, capacity = [
                int(j) for j in input_source.next_line().split()
            ]
            transport_lines.append(
                TransportLine(building_id_1, building_id_2, capacity)
            )
        self.transport_lines = transport_lines

        num_pods = int(input_source.next_line())
        pods = []
        for i in range(num_pods):
            pod_properties = input_source.next_line().split()
            pod_id = int(pod_properties[0])
            itinerary = [int(item) for item in pod_properties[2:]]
            pods.append(Pod(pod_id, itinerary))
        self.pods = pods

        num_new_buildings = int(input_source.next_line())
        for i in range(num_new_buildings):
            module_properties = input_source.next_line().split()
            building_type = int(module_properties[0])
            module_id = int(module_properties[1])
            coordinates = int(module_properties[2]), int(module_properties[3])
            module: Module
            if building_type == LANDING_PAD_BUILDING_TYPE:
                # int(module_properties[4])  # num_astronauts
                astronauts = [
                    Astronaut(int(a_type)) for a_type in module_properties[5:]
                ]
                module = LandingPad(
                    building_type,
                    module_id,
                    coordinates,
                    astronauts,
                )
            else:
                module = Module(building_type, module_id, coordinates)

            self.buildings.append(module)


        for building in self.buildings:
            self.buildings_by_type.setdefault(building.type, []).append(building)

        return num_new_buildings

    def get_building_coordinates(self) -> np.ndarray:
        return np.array(
            [building.coordinates for building in self.buildings], dtype=np.float64
        )


class ActionType(Enum):
    TUBE = "TUBE"
    UPGRADE = "UPGRADE"
    TELEPORT = "TELEPORT"
    POD = "POD"
    DESTROY = "DESTROY"
    WAIT = "WAIT"


class LunarDayAction:
    def __init__(self, type: ActionType) -> None:
        self.type = type


def transport_line_cost(
    coord1: tuple[float, float] | np.ndarray, coord2: tuple[float, float] | np.ndarray
) -> int:
    return int(np.floor(10 * np.linalg.norm(np.array(coord1) - np.array(coord2))))


class CreateTube(LunarDayAction):
    """
    Cost is 1 resource per 0.1 km rounded down.
    """

    def __init__(self, building_1: Module, building_2: Module) -> None:
        super().__init__(ActionType.TUBE)
        self.building_1 = building_1
        self.building_2 = building_2
        self.building_id_1 = building_1.id
        self.building_id_2 = building_2.id

    def cost(self) -> int:
        cost = transport_line_cost(
            self.building_1.coordinates, self.building_2.coordinates
        )
        return cost

    def __str__(self) -> str:
        return f"{self.type.value} {self.building_id_1} {self.building_id_2}"


class UpgradeTube(LunarDayAction):
    """
    Cost is initial cost * new capacity.
    """

    def __init__(self, building_id_1: BuildingId, building_id_2: BuildingId) -> None:
        super().__init__(ActionType.UPGRADE)
        self.building_id_1 = building_id_1
        self.building_id_2 = building_id_2

    def __str__(self) -> str:
        return f"{self.type.value} {self.building_id_1} {self.building_id_2}"


@dataclass
class Teleport(LunarDayAction):
    """
    Cost is 5000
    """

    def __init__(
        self, building_id_entrance: BuildingId, building_id_exit: BuildingId
    ) -> None:
        super().__init__(ActionType.TELEPORT)
        self.building_id_entrance = building_id_entrance
        self.building_id_exit = building_id_exit

    def __str__(self) -> str:
        return f"{self.type.value} {self.building_id_entrance} {self.building_id_exit}"


@dataclass
class CreatePod(LunarDayAction):
    """
    Cost is 1000
    """

    def __init__(self, pod_id: int, itinerary: list[BuildingId]) -> None:
        super().__init__(ActionType.POD)
        self.pod_id = pod_id
        self.itinerary = itinerary

    def cost(self) -> int:
        return 1000

    def __str__(self) -> str:
        return f"{self.type.value} {self.pod_id} {' '.join([str(building_id) for building_id in self.itinerary])}"


def get_pod_path_coordinates(
    pod: Pod | CreatePod, buildings: list[Module]
) -> np.ndarray:
    return np.array(
        [buildings[building_id].coordinates for building_id in pod.itinerary]
    )


@dataclass
class DestroyPod(LunarDayAction):
    """
    Get 750 resources back
    """

    def __init__(self, pod_id: int) -> None:
        super().__init__(ActionType.DESTROY)
        self.pod_id = pod_id

    def __str__(self) -> str:
        return f"{self.type.value} {self.pod_id}"


@dataclass
class Wait(LunarDayAction):
    type: ActionType = ActionType.WAIT

    def __str__(self) -> str:
        return f"{self.type.value}"


def send_actions(actions: list[LunarDayAction]) -> None:
    if len(actions) == 0:
        send_actions([Wait()])
        return
    print(";".join([f"{str(action)}" for action in actions]))


NO_PREDECESSOR = -9999


@dataclass
class Path:
    nodes: list[int]
    cost: float

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, index: int) -> int:
        return self.nodes[index]

    def __iter__(self) -> Generator[int, None, None]:
        for node in self.nodes:
            yield node


class Graph:
    def __init__(
        self,
        routes_to_build: list[tuple[Module, Module]],
        adjacency_matrix: np.ndarray | None = None,
        dist_matrix: np.ndarray | None = None,
        predecessors: np.ndarray | None = None,
        coordinates: np.ndarray | None = None,
        buildings_by_type: dict[BuildingType, list[Module]] | None = None,
    ) -> None:
        self.graph: scipy.spatial.Delaunay | None = None
        self.adjacency_matrix = (
            adjacency_matrix if adjacency_matrix is not None else np.array([])
        )
        self.dist_matrix = dist_matrix if dist_matrix is not None else np.array([])
        self.predecessors = predecessors if predecessors is not None else np.array([])
        self.coordinates = coordinates if coordinates is not None else np.array([])
        self.routes_to_build = routes_to_build
        self.buildings_by_type = (
            buildings_by_type if buildings_by_type is not None else {}
        )

    def get_landing_pads(self) -> list[LandingPad]:
        return self.buildings_by_type.get(LANDING_PAD_BUILDING_TYPE, [])  # type: ignore

    def get_path(self, origin: int, goal: int) -> Path:
        path = [goal]
        k = goal
        cost = 0.0
        while self.predecessors[origin, k] != NO_PREDECESSOR:
            path.append(self.predecessors[origin, k])
            k = self.predecessors[origin, k]
            cost += self.dist_matrix[k, path[-1]]
        return Path(path[::-1], cost)

    def get_adjacent_buildings(self, building_id: BuildingId) -> list[BuildingId]:
        return [
            i
            for i in range(len(self.adjacency_matrix[building_id]))
            if not np.isinf(self.adjacency_matrix[building_id, i])
        ]

    def find_random_node_of_type(self, origin: int, search_type: BuildingType) -> tuple[BuildingId, float]:
        buildings = self.buildings_by_type.get(search_type, [])
        if len(buildings) == 0:
            return -1, float("inf")
        building = np.random.choice(buildings)  # type: ignore
        return building.id, float(np.linalg.norm(self.coordinates[origin] - building.coordinates))

    def find_nearest_node_of_type(
        self, origin: int, search_type: BuildingType
    ) -> tuple[int, float]:
        nearest_node = -1
        nearest_distance = float("inf")
        buildings = self.buildings_by_type.get(search_type, [])
        for building in buildings:
            distance = float(
                np.linalg.norm(self.coordinates[origin] - building.coordinates)
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node = building.id
        return nearest_node, nearest_distance


class GraphBuilder:
    def __init__(self) -> None:
        pass

    def build_transport_lines(self, data: LunarMonthData) -> Graph:
        coordinates = data.get_building_coordinates()
        if len(coordinates) <= 1:
            return Graph(
                [], coordinates=coordinates, buildings_by_type=data.buildings_by_type
            )
        elif len(coordinates) == 2:
            adjacency_matrix = np.array(
                [
                    [0, transport_line_cost(coordinates[0], coordinates[1])],
                    [transport_line_cost(coordinates[0], coordinates[1]), 0],
                ]
            )
            return Graph(
                [(data.buildings[0], data.buildings[1])],
                adjacency_matrix,
                adjacency_matrix,
                np.array([[NO_PREDECESSOR, 1], [0, NO_PREDECESSOR]]),
                coordinates=coordinates,
                buildings_by_type=data.buildings_by_type,
            )

        self.graph = scipy.spatial.Delaunay(coordinates)
        adjacency_matrix = self.delaunay_to_adjacency_matrix(
            self.graph, transport_line_cost
        )
        dist_matrix, predecessors = shortest_path(
            adjacency_matrix, directed=False, method="auto", return_predecessors=True
        )

        graph = Graph(
            [],
            adjacency_matrix,
            dist_matrix,
            predecessors,
            coordinates,
            data.buildings_by_type,
        )

        route_indices_to_build: set[IndexPair] = set()
        for landing_pad in graph.get_landing_pads():
            assert isinstance(landing_pad, LandingPad)
            paths = self.compute_all_paths(graph, landing_pad.id)
            for path in paths:
                for i in range(len(path) - 1):
                    route_indices_to_build.add(IndexPair(path[i], path[i + 1]))

        graph.routes_to_build = [
            (data.buildings[building_id_1], data.buildings[building_id_2])
            for building_id_1, building_id_2 in route_indices_to_build
        ]
        return graph

    def compute_all_paths(self, graph: Graph, origin_node: int) -> list[Path]:
        paths: list[Path] = []
        for node in range(len(graph.coordinates)):
            if node == origin_node:
                continue
            if np.isinf(graph.dist_matrix[origin_node, node]):
                continue
            path = graph.get_path(origin_node, node)
            paths.append(path)
        return paths

    def delaunay_to_adjacency_matrix(
        self,
        delaunay: scipy.spatial.Delaunay,
        cost_function: Callable[[np.ndarray, np.ndarray], float],
        connection_limit: int = 5,
    ) -> np.ndarray:
        n = len(delaunay.points)
        adjacency_matrix = np.full((n, n), np.inf)
        for simplex in delaunay.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    p1, p2 = simplex[i], simplex[j]
                    cost = cost_function(delaunay.points[p1], delaunay.points[p2])
                    adjacency_matrix[p1][p2] = cost
                    adjacency_matrix[p2][p1] = cost
        for row_index in range(len(adjacency_matrix)):
            row = adjacency_matrix[row_index]
            sorted_indices = np.argsort(row)
            culled_connections = sorted_indices[connection_limit:]
            adjacency_matrix[row_index, culled_connections] = np.inf
            adjacency_matrix[culled_connections, row_index] = np.inf

        return adjacency_matrix


def create_all_possible_tubes(
    remaining_resources: int,
    routes_to_build: list[tuple[Module, Module]],
) -> tuple[list[LunarDayAction], int]:
    actions: list[LunarDayAction] = []
    for route in routes_to_build:
        action = CreateTube(route[0], route[1])
        cost = action.cost()
        if remaining_resources < cost:
            break
        remaining_resources -= cost
        actions.append(action)
    return actions, remaining_resources


def build_node_sequence(
    graph: Graph,
    search_types: list[BuildingType],
    origin_id: BuildingId,
    node_sequence: list[BuildingId] = [],
) -> list[int]:
    if len(node_sequence) == 0:
        node_sequence.append(origin_id)
    if len(search_types) == 0:
        return node_sequence
    next_id = -1
    next_distance = float("inf")
    selected_type = 0
    for building_type in search_types:
        nearest_id, nearest_distance = graph.find_nearest_node_of_type(
            origin_id, building_type
        )
        if nearest_id != -1 and nearest_distance < next_distance:
            next_id = nearest_id
            next_distance = nearest_distance
            selected_type = building_type
    if next_id == -1:
        return node_sequence
    node_sequence.append(next_id)
    search_types.remove(selected_type)
    sequence = build_node_sequence(graph, search_types, next_id, node_sequence)
    return sequence


def next_pod_id(pods: list[Pod]) -> int:
    return max([pod.pod_id for pod in pods], default=-1) + 1


def build_path_from_sequence(
    graph: Graph, node_sequence: list[BuildingId]
) -> Path:
    full_path = Path([], 0.0)
    for prev_id, next_id in zip(node_sequence[:-1], node_sequence[1:]):
        path = graph.get_path(prev_id, next_id)
        if len(full_path) == 0:
            nodes = path.nodes
        else:
            nodes = path.nodes[1:]
        full_path.nodes.extend(nodes)
        full_path.cost += path.cost
    return full_path


def create_all_possible_pods(
    remaining_resources: int, graph: Graph, existing_pods: list[Pod]
) -> tuple[list[CreatePod], int]:
    actions: list[CreatePod] = []
    pod_id = next_pod_id(existing_pods)
    for landing_pad in graph.get_landing_pads():
        # node_sequence = build_node_sequence(
        #     graph, list(landing_pad.astronauts_type_map.keys()), landing_pad.id
        # )
        adjacent_buildings = graph.get_adjacent_buildings(landing_pad.id)
        node_sequence = []
        for building_id in adjacent_buildings:
            node_sequence.append(landing_pad.id)
            node_sequence.append(building_id)
        path = build_path_from_sequence(graph, node_sequence)
        action = CreatePod(pod_id, path.nodes)
        pod_id += 1
        cost = action.cost()
        if remaining_resources < cost:
            break
        remaining_resources -= cost
        actions.append(action)
    return actions, remaining_resources


def main() -> None:
    tick = 0
    schedule = []

    data = LunarMonthData()
    graph_builder = GraphBuilder()
    graph: Graph = Graph([], coordinates=np.array([]), buildings_by_type={})
    unbuilt_routes: list[tuple[Module, Module]] = []
    tick_start_time = 0.0

    def time_remaining() -> float:
        if tick == 0:
            time_limit = 1.0
        else:
            time_limit = 0.5
        return time_limit - (time.perf_counter() - tick_start_time)

    while True:
        tick_start_time = time.perf_counter()
        num_new_buildings = data.update_from_input(LiveInputSource())
        debug_print(f"Tick {tick}, resources: {data.resources}")
        # debug_print(data.to_compressed_string())

        actions = []

        remaining_resources = data.resources
        if num_new_buildings > 0:
            graph = graph_builder.build_transport_lines(data)
            tube_actions, remaining_resources = create_all_possible_tubes(
                remaining_resources, graph.routes_to_build
            )
            actions.extend(tube_actions)
            unbuilt_routes.extend(graph.routes_to_build)

            debug_print(
                f"Remaining resources: {remaining_resources}. {len(graph.routes_to_build)} routes left to build."
            )
        else:
            debug_print(f"attempting to build {len(unbuilt_routes)} routes")
            tube_actions, remaining_resources = create_all_possible_tubes(
                remaining_resources, unbuilt_routes
            )
            actions.extend(tube_actions)

        if remaining_resources > 0 and time_remaining() > 0.1:
            debug_print(f"pods: {data.pods}: time_remaining(): {time_remaining()}")
            pod_actions, remaining_resources = create_all_possible_pods(
                remaining_resources, graph, data.pods
            )
            actions.extend(pod_actions)

        if remaining_resources < 0:
            debug_print(f"Not enough resources for {actions.pop()}")

        schedule.append(actions)

        try:
            actions = schedule.pop(0)
        except IndexError:
            actions = []
        send_actions(actions)
        tick += 1


if __name__ == "__main__":
    main()
