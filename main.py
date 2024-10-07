from __future__ import annotations
import math
import sys
import scipy.spatial
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np


def debug_print(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


BuildingId = int
AstronautType = int
BuildingType = int

LANDING_PAD_BUILDING_TYPE = 0
POD_COST = 1000


@dataclass
class TransportLine:
    building_id_1: BuildingId
    building_id_2: BuildingId
    capacity: int


@dataclass
class Pod:
    pod_id: int
    itinerary: list[BuildingId]
    capacity: int = 10


@dataclass
class Module:
    type: BuildingType
    id: BuildingId
    coordinates: tuple[int, int]


@dataclass
class Astronaut:
    type: AstronautType


@dataclass
class LandingPad(Module):
    astronauts: list[Astronaut]
    astronauts_type_map: dict[AstronautType, list[Astronaut]]


class InputSource(ABC):
    def next_line(self) -> str: ...


class LiveInputSource(InputSource):
    def next_line(self) -> str:
        return input()


class BufferedInputSource(InputSource):
    def __init__(self) -> None:
        self.buffer = []

    def append(self, line: str) -> str:
        self.buffer.append(line)
        return line

    def next_line(self) -> str:
        return self.buffer.pop(0)


class LunarMonthData:
    def __init__(self) -> None:
        self.resources = 0
        self.transport_lines: list[TransportLine] = []
        self.pods: list[Pod] = []

        self.buildings: list[Module] = []
        self.buildings_type_map: dict[BuildingType, list[Module]] = {}

    def update_from_input(self, input_source: InputSource) -> None:
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
            if building_type == LANDING_PAD_BUILDING_TYPE:
                num_astronauts = int(module_properties[4])
                astronauts = [
                    Astronaut(int(a_type)) for a_type in module_properties[5:]
                ]
                astronauts_type_map: dict[AstronautType, list[Astronaut]] = {}
                for astronaut in astronauts:
                    if astronaut.type not in astronauts_type_map:
                        astronauts_type_map[astronaut.type] = []
                    astronauts_type_map[astronaut.type].append(astronaut)
                module = LandingPad(
                    building_type,
                    module_id,
                    coordinates,
                    astronauts,
                    astronauts_type_map,
                )
            else:
                module = Module(building_type, module_id, coordinates)

            self.buildings.append(module)
            if building_type not in self.buildings_type_map:
                self.buildings_type_map[building_type] = []
            self.buildings_type_map[building_type].append(module)

            self.transport_lines = transport_lines
            self.resources = resources
            self.pods = pods


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
        x1, y1 = self.building_1.coordinates
        x2, y2 = self.building_2.coordinates
        distance = math.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2))
        debug_print(
            f"Distance: {distance} between {self.building_1.id} and {self.building_2.id} "
            f"with coordinates {self.building_1.coordinates} and {self.building_2.coordinates}"
        )
        return math.floor(distance * 0.1)

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


def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


def build_adjacency_matrix(coordinates: np.ndarray) -> np.ndarray:
    n = len(coordinates)
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            adjacency_matrix[i][j] = dist
            adjacency_matrix[j][i] = dist

    return adjacency_matrix

def main() -> None:
    tick = 0
    schedule = []

    data = LunarMonthData()

    while True:
        data.update_from_input(LiveInputSource())
        debug_print(f"Tick {tick}")
        debug_print(data.resources)
        debug_print(data.transport_lines)

        actions = []

        if tick == 0:
            coordinates = np.array(
                [building.coordinates for building in data.buildings], dtype=np.float64
            )
            graph = scipy.spatial.Delaunay(coordinates)

            debug_print(f"graph {graph.simplices}")

            # landing_pads = data.buildings_type_map[LANDING_PAD_BUILDING_TYPE]

            # # map of landing pad id to list of buildings its astronauts want to visit
            # target_buildings_map: dict[BuildingId, list[Module]] = {}

            # routes_to_build: list[tuple[Module, Module]] = []
            # for landing_pad in landing_pads:
            #     assert isinstance(landing_pad, LandingPad)
            #     target_buildings_map[landing_pad.id] = []
            #     for (
            #         astronaut_type,
            #         astronauts,
            #     ) in landing_pad.astronauts_type_map.items():
            #         target_buildings_map[landing_pad.id].extend(
            #             data.buildings_type_map[astronaut_type]
            #         )
            #         for building in data.buildings_type_map[astronaut_type]:
            #             routes_to_build.append((landing_pad, building))

            # debug_print(target_buildings_map)
            # debug_print(routes_to_build)

            # remaining_resources = data.resources
            # for idx, route in enumerate(routes_to_build):
            #     action = CreateTube(route[0], route[1])
            #     actions.append(action)
            #     remaining_resources = remaining_resources - action.cost()

            # debug_print(f"Remaining resources: {remaining_resources}")

            # start with one pod
            # if remaining_resources >= POD_COST:
            #     itenerary = []
            #     pod_action = CreatePod(1, itenerary)
            #     remaining_resources = remaining_resources - POD_COST
            #     actions.append(pod_action)

        schedule.append(actions)

        try:
            actions = schedule.pop(0)
        except IndexError:
            actions = []
        send_actions(actions)
        tick += 1


if __name__ == "__main__":
    main()
