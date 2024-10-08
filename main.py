from __future__ import annotations
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
AstronautType = int
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
    type: AstronautType

    def to_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Astronaut:
        return cls(int(data["type"]))


@dataclass
class LandingPad(Module):
    astronauts: list[Astronaut]

    def __post_init__(self) -> None:
        self.astronauts_type_map: dict[AstronautType, list[Astronaut]] = {
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
        resources = int(input_source.next_line())
        num_travel_routes = int(input_source.next_line())
        transport_lines = []
        for i in range(num_travel_routes):
            building_id_1, building_id_2, capacity = [
                int(j) for j in input_source.next_line().split()
            ]
            transport_lines.append(
                TransportLine(building_id_1, building_id_2, capacity)
            )

        num_pods = int(input_source.next_line())
        pods = []
        for i in range(num_pods):
            pod_properties = input_source.next_line().split()
            pod_id = int(pod_properties[0])
            itinerary = [int(item) for item in pod_properties[2:]]
            pods.append(Pod(pod_id, itinerary))

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

            self.transport_lines = transport_lines
            self.resources = resources
            self.pods = pods

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
        debug_print(
            f"Line between {self.building_1.id} and {self.building_2.id} "
            f"with coordinates {self.building_1.coordinates} and {self.building_2.coordinates} costs {cost}"
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


class GraphBuilder:
    def __init__(self) -> None:
        self.unbuilt_routes: list[tuple[Module, Module]] = []
        self.graph: scipy.spatial.Delaunay | None = None
        self.adjacency_matrix: np.ndarray | None = None
        self.dist_matrix: np.ndarray | None = None
        self.predecessors: np.ndarray | None = None

    def build_transport_lines(
        self, data: LunarMonthData
    ) -> list[tuple[Module, Module]]:
        coordinates = data.get_building_coordinates()
        if len(coordinates) <= 1:
            return []
        elif len(coordinates) == 2:
            self.adjacency_matrix = np.array(
                [
                    [0, transport_line_cost(coordinates[0], coordinates[1])],
                    [transport_line_cost(coordinates[0], coordinates[1]), 0],
                ]
            )
            self.dist_matrix = self.adjacency_matrix
            self.predecessors = np.array([[NO_PREDECESSOR, 1], [0, NO_PREDECESSOR]])
            return [(data.buildings[0], data.buildings[1])]

        self.graph = scipy.spatial.Delaunay(coordinates)
        adjacency_matrix = self.delaunay_to_adjacency_matrix(
            self.graph, transport_line_cost
        )
        landing_pads = data.buildings_by_type[LANDING_PAD_BUILDING_TYPE]
        dist_matrix, predecessors = shortest_path(
            adjacency_matrix, directed=False, method="auto", return_predecessors=True
        )
        self.dist_matrix = dist_matrix
        self.predecessors = predecessors
        self.adjacency_matrix = adjacency_matrix

        route_indices_to_build: set[IndexPair] = set()
        for landing_pad in landing_pads:
            assert isinstance(landing_pad, LandingPad)
            self.paths = self.compute_all_paths(
                coordinates, dist_matrix, predecessors, landing_pad.id
            )
            for path in self.paths:
                for i in range(len(path) - 1):
                    route_indices_to_build.add(IndexPair(path[i], path[i + 1]))

        routes_to_build = [
            (data.buildings[building_id_1], data.buildings[building_id_2])
            for building_id_1, building_id_2 in route_indices_to_build
        ]
        return routes_to_build

    def compute_all_paths(
        self,
        coordinates: np.ndarray,
        dist_matrix: np.ndarray,
        predecessors: np.ndarray,
        origin_node: int,
    ) -> list[list[int]]:
        paths = []
        for i in range(len(coordinates)):
            if i == origin_node:
                continue
            if np.isinf(dist_matrix[origin_node, i]):
                continue
            path = self.get_path(predecessors, origin_node, i)
            paths.append(path)
        return paths

    def get_path(self, predecessors: np.ndarray, origin: int, goal: int) -> list[int]:
        path = [goal]
        k = goal
        while predecessors[origin, k] != NO_PREDECESSOR:
            path.append(predecessors[origin, k])
            k = predecessors[origin, k]
        return path[::-1]

    def delaunay_to_adjacency_matrix(
        self,
        delaunay: scipy.spatial.Delaunay,
        cost_function: Callable[[np.ndarray, np.ndarray], float],
    ) -> np.ndarray:
        n = len(delaunay.points)
        adjacency_matrix = np.full((n, n), np.inf)
        for simplex in delaunay.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    p1, p2 = simplex[i], simplex[j]
                    dist = cost_function(delaunay.points[p1], delaunay.points[p2])
                    adjacency_matrix[p1][p2] = dist
                    adjacency_matrix[p2][p1] = dist

        return adjacency_matrix


def create_all_possible_tubes(
    remaining_resources: int,
    routes_to_build: list[tuple[Module, Module]],
) -> list[LunarDayAction]:
    actions: list[LunarDayAction] = []
    for route in routes_to_build:
        action = CreateTube(route[0], route[1])
        cost = action.cost()
        if remaining_resources < cost:
            break
        remaining_resources -= cost
        actions.append(action)
    return actions


def main() -> None:
    tick = 0
    schedule = []

    data = LunarMonthData()
    graph_builder = GraphBuilder()

    while True:
        num_new_buildings = data.update_from_input(LiveInputSource())
        debug_print(f"Tick {tick}")
        debug_print(data.to_compressed_string())

        actions = []

        remaining_resources = data.resources
        if num_new_buildings > 0:
            routes_to_build = graph_builder.build_transport_lines(data)
            tube_actions = create_all_possible_tubes(
                remaining_resources, routes_to_build
            )
            actions.extend(tube_actions)
            graph_builder.unbuilt_routes = routes_to_build

            debug_print(
                f"Remaining resources: {remaining_resources}. {len(routes_to_build)} routes left to build."
            )
        else:
            tube_actions = create_all_possible_tubes(
                remaining_resources, graph_builder.unbuilt_routes
            )

        while remaining_resources > 0:
            itinerary = [0, 1, 0, 2, 0, 1, 0, 2]
            pod_action = CreatePod(1, itinerary)
            remaining_resources = remaining_resources - POD_COST
            actions.append(pod_action)

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
