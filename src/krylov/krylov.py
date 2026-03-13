import os
import time
from enum import Enum

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eig

from utils.img_utils import load_image, create_image_from_graph
from utils.graph_utils import convert_to_graph

class Mode(Enum):
    GRAM_SCHMIDT = "gram_schmidt"
    LANCZOS = "lanczos"

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

def gram_schmidt(N, v0, B):
    V_TOT = np.zeros((N, N))
    V_TOT[:, 0] = v0

    V_TOT = np.zeros((N, N))
    V_TOT[:, 0] = v0
    for i in range(1, N):
        w = B @ V_TOT[:, i-1]
        for j in range(i):
            w -= np.dot(V_TOT[:, j], w) * V_TOT[:, j]
        V_TOT[:, i] = w / np.linalg.norm(w)

    return V_TOT

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

def krylov_reconstruction(B, min_M, max_M, mode):
    start_time = time.perf_counter()

    energies = []
    divisions = []

    V_TOT = None

    if mode == Mode.GRAM_SCHMIDT:
        N = len(B)

        v0 = np.random.rand(N)
        v0 = v0 / np.linalg.norm(v0)

        V_TOT = gram_schmidt(N, v0, B)

    for M in range(min_M, max_M + 1):
        energy, division = krylov_iteration(B, M, V_TOT)
        energies.append(energy)
        divisions.append(division)

    end_time = time.perf_counter()
    print(f"Krylov reconstruction took {end_time - start_time:.4f} seconds")

    return {
        "Energies": energies,
        "Reconstructed divisions": divisions,
    }

def calculate_average_results(B, min_M, max_M, minimum=True, trials=50, mode=Mode.LANCZOS, size=None):
    M = max_M - min_M + 1
    sign = -1 if minimum else 1

    true_solution = leading_eigenvector(B)
    energy_true = sign * np.array(true_solution["Energy"])

    energies = np.zeros(M)
    best_energy = -sign * (10**10)
    best_energies = np.zeros(M)
    communities = None

    for i in range(trials):
        t1 = time.perf_counter()
        approx_solution = krylov_reconstruction(B, min_M, max_M, mode)
        t2 = time.perf_counter()

        print(f"[{mode.name} | i={i}] Krylov reconstruction took {t2 - t1:.4f} seconds")

        with open(f"results.csv", "a") as f:
            f.write(f"{size};{mode.name};{t2 - t1:.4f}\n")

        energy = sign * np.array(approx_solution["Energies"])
        energies += energy

        if minimum:
            idx_best = np.argmin(energy)
            current_best_energy = energy[idx_best]
            if current_best_energy < best_energy:
                best_energy = current_best_energy
                best_energies = energy
                communities = approx_solution["Reconstructed divisions"][idx_best]
        else:
            idx_best = np.argmax(energy)
            current_best_energy = energy[idx_best]
            if current_best_energy > best_energy:
                best_energy = current_best_energy
                best_energies = energy
                communities = approx_solution["Reconstructed divisions"][idx_best]

    energies /= trials

    return energies, energy_true, best_energies, communities


# -------------------------------------------------
# Plot
# -------------------------------------------------

def plot_average_results(min_M, max_M, energies, energy_true, best_energies):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.plot(range(min_M, max_M + 1), energies, '-o', markersize=2, label="Average")
    ax.plot(range(min_M, max_M + 1), best_energies, '-x', markersize=2, label='Best trial')
    ax.plot([0, max_M], [energy_true, energy_true], '--', linewidth=1, label='True energy')
    ax.plot([0, max_M], [0, 0], '-', linewidth=.5)

    ax.set_xlabel("M")
    ax.set_ylabel("Energy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Energy vs Krylov subspace dimension")

    fig.tight_layout()
    return fig, ax

def plot_images(original_img, segmented_img):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(original_img)
    axs[0].axis('off')
    axs[0].set_title('Original')

    axs[1].imshow(segmented_img)
    axs[1].axis('off')
    axs[1].set_title('Segmented')

def generate_synthetic_image(size):
    img = np.zeros((size, size, 3), dtype=np.uint8)

    img[::2, :] = np.random.randint(0, 256, size=3)
    img[1::2,:] = np.random.randint(0, 256, size=3)

    return img

# -------------------------------------------------
# Main segmentation
# -------------------------------------------------

def segment_image(img, resolution=1, beta=100, min_M=1, max_M=20, minimum=False, trials=50, mode=Mode.LANCZOS):
    graph = convert_to_graph(img, beta)

    B, _ = create_B(graph, resolution=resolution, norm=False)
    print(B.shape)

    energies, energy_true, best_energies, communities = calculate_average_results(
        B, min_M, max_M, minimum, trials, mode, size=img.shape[0]
    )

    fig, ax = plot_average_results(min_M, max_M, energies, energy_true, best_energies)

    for community in communities:
        community_color = np.random.randint(0, 255, size=3)
        for node in community:
            graph.nodes[node]['color'] = community_color

    new_img = create_image_from_graph(graph, img.shape[:2])

    return new_img, fig, ax


# -------------------------------------------------
# Run
# -------------------------------------------------

if __name__ == "__main__":
    # img_name = "80x80_low_contrast_blue.png"
    # path = os.path.join(os.path.dirname(__file__), "../img", "two_comm", img_name)

    # img_name = "two_comms_40x40.png"
    # path = os.path.join(os.path.dirname(__file__), "../img", img_name)

    # img_name = "cube.png"
    # path = os.path.join(os.path.dirname(__file__), "../img", "two_comm", img_name)

    # img = load_image(path)
    # segmented_img, energy_fig, energy_ax = segment_image(img, trials=1, mode=Mode.GRAM_SCHMIDT)

    SIZES = [20, 40, 60, 80, 100, 120]

    for size in SIZES:
        img = generate_synthetic_image(size)
        segmented_img, energy_fig, energy_ax = segment_image(img, trials=1, mode=Mode.LANCZOS)
        segmented_img, energy_fig, energy_ax = segment_image(img, trials=1, mode=Mode.GRAM_SCHMIDT)

    # plot_images(img, segmented_img)
    # plt.show()