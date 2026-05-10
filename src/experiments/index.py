import os
import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eig
from sklearn.metrics import normalized_mutual_info_score

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from dwave.embedding.zephyr import find_clique_embedding

from utils.img_utils import load_image, create_image_from_graph
from utils.graph_utils import convert_to_graph
from utils.file_utils import maybe_create_output_dir

from common import Mode, get_results_path, get_image_path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def leading_eigenvector(B):
    eigvals_B, eigvecs_B = eig(B)

    eigvals_B = eigvals_B.real
    eigvecs_B = eigvecs_B.real

    idx_sorted = np.argsort(eigvals_B)
    v_max = eigvecs_B[:, idx_sorted[-1]]

    return v_max

def create_B(G, resolution=1):
    A = nx.to_numpy_array(G)
    g = A.sum(axis=1)
    m = 0.5 * g.sum()

    P = resolution * np.outer(g, g) / (2 * m)
    B = A - P

    return B

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

        v_new = w / beta[j - 1]
        V[:, j] = v_new

        w = B @ v_new - beta[j - 1] * v
        alpha[j] = np.dot(v_new, w)
        w = w - alpha[j] * v_new

        v = v_new

    return V, alpha, beta

def gram_schmidt(B, v0, M):
    N = len(B)
    V_TOT = np.zeros((N, M))
    V_TOT[:, 0] = v0
    for i in range(1, M):
        w = B @ V_TOT[:, i-1]
        for j in range(i):
            w -= np.dot(V_TOT[:, j], w) * V_TOT[:, j]
        V_TOT[:, i] = w / np.linalg.norm(w)

    return V_TOT

def reduce_modularity_matrix(B, V_TOT, M, mode):
    B_red = None
    V = None

    if mode == Mode.LANCZOS:
        V_full, alpha_full, beta_full = V_TOT
        V = V_full[:, :M]
        alpha = alpha_full[:M]
        beta = beta_full[:M - 1]
        B_red = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    else:
        V = V_TOT[:, :M]
        B_red = V.T @ B @ V

    orthogonality = np.sum(V.T @ V)
    
    return B_red, V, orthogonality

def construct_division(B, V, v_max_red_prime):
    vr = V @ v_max_red_prime
    vr -= np.mean(vr)
    vr /= np.linalg.norm(vr)

    partition = np.where(vr > 0, 1, -1)
    division = [np.where(partition == -1)[0].tolist(), np.where(partition == 1)[0].tolist()]
    energy = partition.T @ B @ partition

    return vr, division, energy

def krylov_iteration_dwave(B, V_TOT,M, mode):
    B_red, V, orthogonality = reduce_modularity_matrix(B, V_TOT, M, mode)
    h = defaultdict(int)
    J = -B_red

    sampler = DWaveSampler("Advantage2_system4.3")
    embedding = find_clique_embedding(
        B_red.shape[0],
        target_graph=sampler.to_networkx_graph()
    )

    sampler = FixedEmbeddingComposite(sampler, embedding=embedding)

    start_time = time.perf_counter()
    response = sampler.sample_ising(h, J, num_reads=100)
    response.resolve()
    end_time = time.perf_counter()

    problem_id = response.info['problem_id']

    v_max_red_prime = np.array(
        [val for val in response.first.sample.values()]
    )

    vr, division, energy = construct_division(B, V, v_max_red_prime)

    return {
        "division": division,
        "leading_v": vr,
        "orthogonality": orthogonality,
        "energy": energy,
        "problem_id": problem_id,
        "dwave_call_time": end_time - start_time
    }

def krylov_iteration_eigensolver(B, V_TOT, M, mode):
    B_red, V, orthogonality = reduce_modularity_matrix(B, V_TOT, M, mode)
    
    eigvals_red, eigvecs_red = eig(B_red)
    eigvals_red = eigvals_red.real
    eigvecs_red = eigvecs_red.real

    idx = np.argsort(eigvals_red)
    v_max_red = eigvecs_red[:, idx[-1]]

    leading_v = V @ v_max_red
    leading_v_shifted = leading_v - np.mean(leading_v)

    leading_v /= np.linalg.norm(leading_v)
    leading_v_shifted /= np.linalg.norm(leading_v_shifted)

    _, division, energy = construct_division(B, V, v_max_red)

    return {
        "division": division,
        "energy": energy,
        "leading_v": leading_v,
        "leading_v_shifted": leading_v_shifted,
        "orthogonality": orthogonality
    }

def krylov_iteration(B, M, V_TOT, mode, use_dwave):
    if use_dwave:
        return krylov_iteration_dwave(B, V_TOT, M, mode)
    else:
        return krylov_iteration_eigensolver(B, V_TOT, M, mode)

def krylov_reconstruction(B, min_M, max_M, mode, use_dwave, use_normal_distribution=False):
    data = defaultdict()

    N = len(B)

    v0 = np.random.normal(size=N) if use_normal_distribution else np.random.rand(N)
    v0 /= np.linalg.norm(v0)

    base_generator = gram_schmidt if mode == Mode.GRAM_SCHMIDT else lanczos
    V_TOT = base_generator(B, v0, max_M)

    for M in range(min_M, max_M + 1):
        data[M] = krylov_iteration(B, M, V_TOT, mode, use_dwave=use_dwave)

    return {
        "v0": v0,
        "M": data
    }

def plot_images(original_img, segmented_img, img_name, mode, use_dwave):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(original_img)
    axs[0].axis('off')
    axs[0].set_title('Original image')

    axs[1].imshow(segmented_img)
    axs[1].axis('off')
    axs[1].set_title("Segmented image")

    filename = f"{img_name}_{mode.value}_{'annealer' if use_dwave else 'eigensolver'}.png"
    path = os.path.join(get_results_path(img_name), filename)

    plt.savefig(path)

def ground_truth_communities(img):
    communities = {}

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            node_id = y * img.shape[1] + x
            color = img[y][x]
            community_key = f"{color[0]}_{color[1]}_{color[2]}"

            if community_key not in communities:
                communities[community_key] = set()

            communities[community_key].add(node_id)
            
    return list(communities.values())

def prepare_ground_truth(img, modularity_matrix, graph):
    true_leading_v = leading_eigenvector(modularity_matrix)
    true_segmentation = ground_truth_communities(img)
    true_modularity = nx.algorithms.community.modularity(graph, true_segmentation)

    return {
        "communities": true_segmentation,
        "leading_v": true_leading_v,
        "modularity": true_modularity
    }

def postprocess_data(reconstruction_data, G, ground_truth):
    reconstruction_data["min_M"] = min(int(k) for k in reconstruction_data["M"].keys())
    reconstruction_data["max_M"] = max(int(k) for k in reconstruction_data["M"].keys())
    reconstruction_data["best_M"] =  max(reconstruction_data["M"], key=lambda m: reconstruction_data["M"][m]["energy"])
    reconstruction_data["ground_truth_modularity"] = ground_truth["modularity"]

    all_nodes = set()
    for comm in ground_truth["communities"]:
        all_nodes.update(comm)
    n_nodes = max(all_nodes) + 1 if all_nodes else 0

    gt_labels = np.zeros(n_nodes, dtype=int)
    for label, comm in enumerate(ground_truth["communities"]):
        for node in comm:
            gt_labels[node] = label

    for data in reconstruction_data["M"].values():
        data["modularity"] = nx.algorithms.community.modularity(G, data["division"])
        data["cosine_similarity"] = np.abs(np.dot(ground_truth["leading_v"].conj(), data["leading_v"]))
        if "leading_v_shifted" in data:
            data["cosine_similarity_shifted"] = np.abs(np.dot(ground_truth["leading_v"].conj(), data["leading_v_shifted"]))

        comm_labels = np.zeros(n_nodes, dtype=int)
        for label, comm in enumerate(data["division"]):
            for node in comm:
                comm_labels[node] = label
        data["NMI"] = normalized_mutual_info_score(gt_labels, comm_labels)

def generate_segmented_image(graph, communities):
    segmented_graph = graph.copy()
    for community in communities:
        community_color = np.random.randint(0, 255, size=3)
        for node in community:
            segmented_graph.nodes[node]['color'] = community_color

    new_img = create_image_from_graph(segmented_graph, img.shape[:2])

    return new_img

def save_to_file(img_name, mode, use_dwave, data):
    filename = f"{img_name}_{mode.value}_{'annealer' if use_dwave else 'eigensolver'}.json"
    path = os.path.join(get_results_path(img_name), filename)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, cls=NumpyEncoder)

BETA = 100
RESOLUTION = 1
MIN_M = 2
MAX_M = 20

MODES = [Mode.LANCZOS, Mode.GRAM_SCHMIDT]
# IMAGES = ["bubbles", "window", "maze", "desert"]
IMAGES = ["window", "maze", "desert"]
USE_DWAVE_OPTIONS = [True, False]

# IMAGES = ["7"]
# MODES = [Mode.GRAM_SCHMIDT]
# USE_DWAVE_OPTIONS = [False]

if __name__ == "__main__":
    for img_name in IMAGES:
        img_path = get_image_path(img_name)
        img = load_image(img_path)

        graph = convert_to_graph(img, BETA)
        modularity_matrix = create_B(graph, resolution=RESOLUTION)
        ground_truth = prepare_ground_truth(img, modularity_matrix, graph)

        maybe_create_output_dir(get_results_path(img_name))

        for mode in MODES:
            for use_dwave in USE_DWAVE_OPTIONS:
                # Convert image to graph and create modularity matrix
                reconstruction_data = krylov_reconstruction(modularity_matrix, MIN_M, MAX_M, mode, use_dwave)
                postprocess_data(reconstruction_data, graph, ground_truth)
                save_to_file(img_name, mode, use_dwave, reconstruction_data)

                best_M = reconstruction_data["best_M"]
                communities = reconstruction_data["M"][best_M]["division"]
                segmented_img = generate_segmented_image(graph, communities)

                plot_images(img, segmented_img, img_name, mode, use_dwave)
