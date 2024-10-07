import scipy.spatial
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path


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


coordinates = np.array(
    [
        (138, 27),
        (132, 30),
        (144, 30),
        (138, 33),
        (132, 36),
        (144, 36),
        (138, 39),
    ],
    dtype=np.float64,
)

adjacency_matrix = build_adjacency_matrix(coordinates)
# graph = scipy.spatial.Delaunay(coordinates)
# plt.triplot(coordinates[:, 0], coordinates[:, 1], graph.simplices)

D, Pr = shortest_path(
    adjacency_matrix, directed=False, method="FW", return_predecessors=True
)

origin_node = 0

plt.plot(coordinates[:, 0], coordinates[:, 1], "o")
for i in range(len(coordinates)):
    plt.text(coordinates[i][0], coordinates[i][1], f"{i}")

for i in range(len(coordinates)):
    for j in range(i + 1, len(coordinates)):
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

for i in range(len(coordinates)):
    if i == origin_node:
        continue
    path = get_path(Pr, origin_node, i)
    for j in range(len(path) - 1):
        plt.plot(
            [coordinates[path[j]][0], coordinates[path[j + 1]][0]],
            [coordinates[path[j]][1], coordinates[path[j + 1]][1]],
            "r-",
        )

plt.show()
