import time

import numpy as np
import networkx as nx
from scipy.linalg import eig
import itertools

def pixels_distance(p1, p2):
    return np.sum((p1 - p2) ** 2)

def find_max_distance(img):
    max_distance = -1

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if x > 0:
                dist = pixels_distance(img[y][x], img[y][x - 1])
                if dist > max_distance:
                    max_distance = dist

            if y > 0:
                dist = pixels_distance(img[y][x], img[y - 1][x])
                if dist > max_distance:
                    max_distance = dist

            if x > 0 and y > 0:
                dist = pixels_distance(img[y][x], img[y - 1][x - 1])
                if dist > max_distance:
                    max_distance = dist

            if x < img.shape[1] - 1 and y > 0:
                dist = pixels_distance(img[y][x], img[y - 1][x + 1])
                if dist > max_distance:
                    max_distance = dist

    return max_distance

def edge_weight(p1, p2, max_d, beta):
    return np.exp(-beta * (pixels_distance(p1, p2) / max_d))

def convert_to_graph(img, beta):
    max_distance = find_max_distance(img)

    G = nx.Graph()

    node_id = 0

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            G.add_node(node_id, color=tuple(img[y, x]))

            node_id += 1

    n = len(G.nodes)

    for v, u in itertools.combinations(range(n), 2):
        x1, y1 = v % img.shape[1], v // img.shape[1]
        x2, y2 = u % img.shape[1], u // img.shape[1]

        weight = edge_weight(img[y1, x1], img[y2, x2], max_distance, beta)
        G.add_edge(v, u, weight=weight)

    return G

def create_image_from_graph(graph, size):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    for v in graph.nodes:
        img[v // size[1]][v % size[1]] = graph.nodes[v]['color']

    return img

def leading_eigenvector(B):
    eigvals_B, eigvecs_B = eig(B)

    eigvals_B = eigvals_B.real
    eigvecs_B = eigvecs_B.real

    idx_sorted = np.argsort(eigvals_B)
    v_max = eigvecs_B[:, idx_sorted[-1]]

    partition = np.where(v_max > 0, 1, -1)

    return {
        "Leading eigenvector": v_max,
        "Partition": partition,
        "Division": [
            [i for i, val in enumerate(partition) if val == -1],
            [i for i, val in enumerate(partition) if val == 1],
        ],
        "Energy": partition.T @ B @ partition,
    }

def create_B(G, resolution=1, norm=False):
    A = nx.to_numpy_array(G)
    g = A.sum(axis=1)
    m = 0.5 * g.sum()

    D_inv_sqrt = np.diag(1 / np.sqrt(g + 1e-10))
    P = resolution * np.outer(g, g) / (2 * m)

    B = A - P
    if norm:
        B = D_inv_sqrt @ B @ D_inv_sqrt

    return B, m

def lanczos(B, v0, k):
    n = len(v0)
    V = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k - 1)

    v = v0 / np.linalg.norm(v0)
    V[:, 0] = v

    w = B @ v
    alpha[0] = np.dot(v, w)
    w = w - alpha[0] * v

    for j in range(1, k):
        beta[j - 1] = np.linalg.norm(w)
        if beta[j - 1] < 1e-12:
            return V[:, :j], alpha[:j], beta[:j - 1]

        v_new = w / beta[j - 1]
        V[:, j] = v_new

        w = B @ v_new - beta[j - 1] * v
        alpha[j] = np.dot(v_new, w)
        w = w - alpha[j] * v_new

        v = v_new

    return V, alpha, beta

def krylov_iteration(B, M, V_TOT):
    B_red = None

    if V_TOT is None:
        v0 = np.random.rand(B.shape[0])
        V, _, _ = lanczos(B, v0, M)
        B_red = V.T @ B @ V
    else:
        V = V_TOT[:,:M]
        B_red = V.T @ B @ V

    eigvals_red, eigvecs_red = eig(B_red)
    eigvals_red = eigvals_red.real
    eigvecs_red = eigvecs_red.real

    idx = np.argsort(eigvals_red)
    v_max_red = eigvecs_red[:, idx[-1]]

    v_max_red_prime = np.sign(v_max_red)

    vr = V @ v_max_red_prime
    vr -= np.mean(vr)
    vr /= np.linalg.norm(vr)

    partition = np.where(vr > 0, 1, -1)

    energy = partition.T @ B @ partition

    division = [np.where(partition == -1)[0].tolist(), np.where(partition == 1)[0].tolist()]

    return energy, division

def krylov_reconstruction(B, min_M, max_M):
    energies = []
    divisions = []

    V_TOT = None

    for M in range(min_M, max_M + 1):
        energy, division = krylov_iteration(B, M, V_TOT)
        energies.append(energy)
        divisions.append(division)

    return {
        "Energies": energies,
        "Reconstructed divisions": divisions,
    }

def calculate_average_results(B, min_M, max_M):
    t1 = time.perf_counter()
    _ = krylov_reconstruction(B, min_M, max_M)
    t2 = time.perf_counter()

    return t2 - t1

def generate_synthetic_image(size):
    img = np.zeros((size, size, 3), dtype=np.uint8)

    img[::2, :] = np.random.randint(0, 256, size=3)
    img[1::2,:] = np.random.randint(0, 256, size=3)

    return img

# -------------------------------------------------
# Main segmentation
# -------------------------------------------------

def segment_image(img, resolution=1, beta=100, min_M=1, max_M=20):
    t1 = time.perf_counter()
    graph = convert_to_graph(img, beta)
    t2 = time.perf_counter()

    graph_generation_time = t2 - t1

    B, _ = create_B(graph, resolution=resolution, norm=False)

    krylov_time = calculate_average_results(B, min_M, max_M)

    return krylov_time, graph_generation_time


# -------------------------------------------------
# Run
# -------------------------------------------------

if __name__ == "__main__":
    SIZES = [20, 40, 60, 80, 100]

    for size in SIZES:
        img = generate_synthetic_image(size)
        krylov_time, graph_generation_time = segment_image(img)


        with open(f"results_lanczos.csv", "a") as f:
            f.write(f"{size};{graph_generation_time:.5f};{krylov_time:.5f}\n")