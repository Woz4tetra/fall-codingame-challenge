from __future__ import annotations

import sys
from abc import ABC
from dataclasses import dataclass

BuildingId = int
AstronautType = int


@dataclass
class TransportLine:
    building_id_1: BuildingId
    building_id_2: BuildingId
    capacity: int


@dataclass
class Pod:
    pod_id: int
    itinerary: list[BuildingId]


@dataclass
class Module:
    type: AstronautType
    coordinates: tuple[int, int]


@dataclass
class Astronaut:
    type: AstronautType


LANDING_PAD_ID = 0


@dataclass
class LandingPad(Module):
    astronauts: list[Astronaut]


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
    pod_properties: list[Pod]
    module_properties: list[Module]

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
        all_pod_properties = []
        for i in range(num_pods):
            pod_properties = input_source.next_line().split()
            pod_id = int(pod_properties[0])
            itinerary = [int(item) for item in pod_properties[2:]]
            all_pod_properties.append(Pod(pod_id, itinerary))

        num_new_buildings = int(input_source.next_line())
        all_module_properties = []
        for i in range(num_new_buildings):
            module_properties = input_source.next_line().split()
            astronaut_type = int(module_properties[0])
            coordinates = int(module_properties[1]), int(module_properties[2])
            if astronaut_type == LANDING_PAD_ID:
                astronauts = [
                    Astronaut(int(a_type)) for a_type in module_properties[3:]
                ]
                module = LandingPad(astronaut_type, coordinates, astronauts)
            else:
                module = Module(astronaut_type, coordinates)
            all_module_properties.append(module)
        self = cls(
            resources, transport_lines, all_pod_properties, all_module_properties
        )
        return self


def test_starter_code() -> None:
    buffer = BufferedInputSource()
    resources = int(buffer.append(input()))
    num_travel_routes = int(buffer.append(input()))
    building_connections = []
    for i in range(num_travel_routes):
        building_id_1, building_id_2, capacity = [
            int(j) for j in buffer.append(input()).split()
        ]
        building_connections.append((building_id_1, building_id_2, capacity))
    num_pods = int(buffer.append(input()))
    all_pod_properties = []
    for i in range(num_pods):
        pod_properties = buffer.append(input())
        all_pod_properties.append(pod_properties)
    num_new_buildings = int(buffer.append(input()))
    all_building_properties = []
    for i in range(num_new_buildings):
        building_properties = buffer.append(input())
        all_building_properties.append(building_properties)

    data = LunarMonthData.parse_from_input(buffer)
    assert data.resources == resources
    assert len(data.transport_lines) == len(building_connections)
    assert len(data.pod_properties) == len(all_pod_properties)
    assert len(data.module_properties) == len(all_building_properties)


def main() -> None:
    test_starter_code()


if __name__ == "__main__":
    main()
