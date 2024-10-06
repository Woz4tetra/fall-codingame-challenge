from __future__ import annotations

import sys
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any


def debug_print(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


BuildingId = int
AstronautType = int


@dataclass
class TransportLine:
    building_id_1: BuildingId
    building_id_2: BuildingId
    capacity: int

    def __post_init__(self) -> None:
        debug_print(f"TransportLine: {self.building_id_1} {self.building_id_2} capacity:{self.capacity}")


@dataclass
class Pod:
    pod_id: int
    itinerary: list[BuildingId]

    def __post_init__(self) -> None:
        debug_print(f"Pod: {self.pod_id} itinerary: {self.itinerary}")


@dataclass
class Module:
    type: AstronautType
    id: BuildingId
    coordinates: tuple[int, int]

    def __post_init__(self) -> None:
        debug_print(f"Module: {self.type} coordinates: {self.coordinates}")


@dataclass
class Astronaut:
    type: AstronautType


LANDING_PAD_ID = 0


@dataclass
class LandingPad(Module):
    astronauts: list[Astronaut]

    def __post_init__(self) -> None:
        debug_print(f"LandingPad: {self.type} coordinates: {self.coordinates} astronauts: {self.astronauts}")


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


@dataclass
class LunarMonthData:
    resources: int
    transport_lines: list[TransportLine]
    pods: list[Pod]
    buildings: list[Module]

    @classmethod
    def parse_from_input(cls, input_source: InputSource) -> LunarMonthData:
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
        buildings = []
        for i in range(num_new_buildings):
            module_properties = input_source.next_line().split()
            building_type = int(module_properties[0])
            module_id = int(module_properties[1])
            coordinates = int(module_properties[2]), int(module_properties[3])
            if building_type == LANDING_PAD_ID:
                num_astronauts = int(module_properties[4])
                astronauts = [
                    Astronaut(int(a_type)) for a_type in module_properties[5:]
                ]
                module = LandingPad(building_type, module_id, coordinates, astronauts)
            else:
                module = Module(building_type, module_id, coordinates)
            buildings.append(module)

        self = cls(
            resources, transport_lines, pods, buildings
        )
        return self


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

    def __init__(self, building_id_1: BuildingId, building_id_2: BuildingId) -> None:
        super().__init__(ActionType.TUBE)
        self.building_id_1 = building_id_1
        self.building_id_2 = building_id_2

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


def main() -> None:
    tick = 0
    schedule = [
        [
            CreateTube(0, 1),
            CreateTube(0, 2),
            CreatePod(42, [0, 1, 0, 2, 0, 1, 0, 2, 0, 1]),
        ],
        [],
    ]
    while True:
        data = LunarMonthData.parse_from_input(LiveInputSource())
        debug_print(f"Tick {tick}")
        debug_print(data.resources)
        if data.resources >= 5000 - 750 + 1000:
            send_actions(
                [
                    Teleport(1, 2),
                    DestroyPod(42),
                    CreatePod(42, [0, 1, 0, 1, 0, 1]),
                ]
            )
        else:
            try:
                actions = schedule.pop(0)
            except IndexError:
                actions = []
            send_actions(actions)
        tick += 1


if __name__ == "__main__":
    main()
