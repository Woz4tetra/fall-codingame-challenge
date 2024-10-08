import scipy.spatial
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path
from main import create_all_possible_tubes, GraphBuilder, LunarMonthData

data = LunarMonthData.from_dict(
    {"resources": 11500, "transport_lines": [], "pods": [], "buildings": [{"type": 2, "id": 0, "coordinates": [103, 71]}, {"type": 0, "id": 1, "coordinates": [38, 47], "astronauts": [{"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}]}, {"type": 1, "id": 2, "coordinates": [144, 36]}, {"type": 1, "id": 3, "coordinates": [138, 33]}, {"type": 0, "id": 4, "coordinates": [32, 45], "astronauts": [{"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}]}, {"type": 5, "id": 5, "coordinates": [132, 30]}, {"type": 4, "id": 6, "coordinates": [101, 73]}, {"type": 11, "id": 7, "coordinates": [138, 39]}, {"type": 4, "id": 8, "coordinates": [138, 27]}, {"type": 7, "id": 9, "coordinates": [99, 75]}, {"type": 4, "id": 10, "coordinates": [101, 69]}, {"type": 0, "id": 11, "coordinates": [34, 47], "astronauts": [{"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}, {"type": 4}]}, {"type": 1, "id": 12, "coordinates": [99, 67]}, {"type": 4, "id": 13, "coordinates": [95, 71]}, {"type": 0, "id": 14, "coordinates": [36, 49], "astronauts": [{"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}]}, {"type": 0, "id": 15, "coordinates": [40, 45], "astronauts": [{"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}, {"type": 7}, {"type": 1}]}, {"type": 7, "id": 16, "coordinates": [97, 73]}, {"type": 2, "id": 17, "coordinates": [132, 36]}, {"type": 0, "id": 18, "coordinates": [36, 41], "astronauts": [{"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}, {"type": 1}]}, {"type": 1, "id": 19, "coordinates": [144, 30]}, {"type": 0, "id": 20, "coordinates": [34, 43], "astronauts": [{"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}, {"type": 8}, {"type": 1}]}, {"type": 8, "id": 21, "coordinates": [97, 69]}, {"type": 0, "id": 22, "coordinates": [38, 43], "astronauts": [{"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}, {"type": 4}, {"type": 2}]}]}
)
graph_builder = GraphBuilder()
routes_to_build = graph_builder.build_transport_lines(data)

paths = create_all_possible_tubes(1000000, routes_to_build)
coordinates = data.get_building_coordinates()

plt.plot(coordinates[:, 0], coordinates[:, 1], "o")
for i in range(len(coordinates)):
    plt.text(coordinates[i][0], coordinates[i][1], f"{i}")

assert graph_builder.adjacency_matrix is not None
adjacency_matrix = graph_builder.adjacency_matrix

for i in range(len(coordinates)):
    for j in range(i + 1, len(coordinates)):
        if np.isinf(adjacency_matrix[i][j]):
            continue
        plt.plot(
            [coordinates[i][0], coordinates[j][0]],
            [coordinates[i][1], coordinates[j][1]],
            "k-",
        )
        plt.text(
            (coordinates[i][0] + coordinates[j][0]) / 2,
            (coordinates[i][1] + coordinates[j][1]) / 2,
            f"{adjacency_matrix[i][j]:.2f}",
        )

assert graph_builder.paths is not None
paths_indices = graph_builder.paths

for path in paths_indices:
    path_cost = 0.0
    for j in range(len(path) - 1):
        path_cost += adjacency_matrix[path[j]][path[j + 1]]
        plt.plot(
            [coordinates[path[j]][0], coordinates[path[j + 1]][0]],
            [coordinates[path[j]][1], coordinates[path[j + 1]][1]],
            "r-",
            label=f"Cost: {path_cost:.2f}",
        )
# for i in range(len(coordinates)):
#     if i == origin_node:
#         continue
#     if np.isinf(D[origin_node, i]):
#         continue
#     path = get_path(Pr, origin_node, i)
#     for j in range(len(path) - 1):
#         plt.plot(
#             [coordinates[path[j]][0], coordinates[path[j + 1]][0]],
#             [coordinates[path[j]][1], coordinates[path[j + 1]][1]],
#             "r-",
#         )

plt.legend()
plt.show()
