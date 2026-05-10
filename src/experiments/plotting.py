import matplotlib.pyplot as plt
import os
import json

from common import Mode, get_results_path
from utils.file_utils import maybe_create_output_dir

def prepare_score_plots(img_name, mode, annealer_data, eigensolver_data, save_plots):
    annealer_color = 'royalblue'
    eigensolver_color = 'limegreen'
    ground_truth_color = 'red'

    min_m = annealer_data["min_M"]
    max_m = annealer_data["max_M"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(range(min_m, max_m+1), annealer_data["modularities"], '-o', markersize=3, label='Annealer', color=annealer_color)
    ax[0].plot(range(min_m, max_m+1), eigensolver_data["modularities"], '-o', markersize=3, label='Eigensolver', color=eigensolver_color)
    ax[0].axhline(y=annealer_data["ground_truth_modularity"], color=ground_truth_color, linestyle='--', label='Ground Truth')
    ax[0].set_xlabel("M")
    ax[0].set_xticks(range(min_m, max_m+1))
    ax[0].set_ylabel("Modularity")
    ax[0].spines[["top", "right"]].set_visible(False)
    ax[0].set_title("Modularity vs Krylov subspace dimension")
    ax[0].legend(loc='lower right')
    ax[0].grid(True)

    ax[1].plot(range(min_m, max_m + 1), annealer_data["NMIs"], '-o', markersize=3, label='Annealer', color=annealer_color)
    ax[1].plot(range(min_m, max_m + 1), eigensolver_data["NMIs"], '-o', markersize=3, label='Eigensolver', color=eigensolver_color)
    ax[1].set_xlabel("M")
    ax[1].set_xticks(range(min_m, max_m + 1))
    ax[1].set_ylabel("Normalized Mutual Information (NMI)")
    ax[1].spines[["top", "right"]].set_visible(False)
    ax[1].set_title("NMI vs Krylov subspace dimension")
    ax[1].legend(loc='lower right')
    ax[1].grid(True)

    fig.tight_layout()

    if save_plots:
        path = get_results_path(img_name)
        plt.savefig(os.path.join(path, "plots", f"{img_name}_{mode.value}_score_plots.png"), dpi=300)

    return fig, ax

def prepare_accuracy_plots(img_name, mode, annealer_data, eigensolver_data, save_plots):
    annealer_color = 'royalblue'
    eigensolver_color = 'limegreen'
    eigensolver_shifted_color = 'forestgreen'
    ground_truth_color = 'red'

    min_m = annealer_data["min_M"]
    max_m = annealer_data["max_M"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(range(min_m, max_m + 1), annealer_data["cosine_similarities"], '-o', markersize=3, label='Annealer', color=annealer_color)
    ax[0].plot(range(min_m, max_m + 1), eigensolver_data["cosine_similarities"], '-o', markersize=3, label='Eigensolver', color=eigensolver_color)
    ax[0].plot(range(min_m, max_m + 1), eigensolver_data["cosine_similarities_shifted"], '-o', markersize=3, label='Eigensolver Shifted', color=eigensolver_shifted_color)
    ax[0].set_xlabel("M")
    ax[0].set_xticks(range(min_m, max_m + 1))
    ax[0].set_ylabel("Cosine similarity with leading eigenvector")
    ax[0].spines[["top", "right"]].set_visible(False)
    ax[0].set_title("Cosine similarity vs Krylov subspace dimension")
    ax[0].legend(loc='lower right')
    ax[0].grid(True)

    ax[1].plot(range(min_m, max_m + 1), annealer_data["orthogonality"], '-o', markersize=3, label='Annealer', color=annealer_color)
    ax[1].plot(range(min_m, max_m + 1), eigensolver_data["orthogonality"], '-o', markersize=3, label='Eigensolver', color=eigensolver_color)
    ax[1].plot(range(min_m, max_m + 1), range(min_m, max_m + 1), '--', color=ground_truth_color, label='Ideal Orthogonality')
    ax[1].set_xlabel("M")
    ax[1].set_xticks(range(min_m, max_m + 1))
    ax[1].set_ylabel("Orthogonality")
    ax[1].spines[["top", "right"]].set_visible(False)
    ax[1].set_title("Orthogonality vs Krylov subspace dimension")
    ax[1].legend(loc='lower right')
    ax[1].grid(True)

    fig.tight_layout()

    if save_plots:
        path = get_results_path(img_name)
        plt.savefig(os.path.join(path, "plots", f"{img_name}_{mode.value}_accuracy_plots.png"), dpi=300)

    return fig, ax

def prepare_plots(img_name, mode, annealer_data, eigensolver_data, save_plots):
    score_fig, score_ax = prepare_score_plots(img_name, mode, annealer_data, eigensolver_data, save_plots)
    accuracy_fig, accuracy_ax = prepare_accuracy_plots(img_name, mode, annealer_data, eigensolver_data, save_plots)

    return (score_fig, score_ax), (accuracy_fig, accuracy_ax)

def load_data(img_name, mode, use_dwave=False):
    filename = f"{img_name}_{mode.value}_{'annealer' if use_dwave else 'eigensolver'}.json"
    path = os.path.join(os.path.dirname(__file__), "results", img_name, filename)

    with open(path, "r") as f:
        return json.load(f)

    return False

def preprocess_data(data):
    m_data = data["M"]

    modularities = []
    cosine_similarities = []
    cosine_similarities_shifted = []
    NMIs = []
    orthogonalities = []

    for M in range(data["min_M"], data["max_M"] + 1):
        modularities.append(m_data[str(M)]["modularity"])
        NMIs.append(m_data[str(M)]["NMI"])
        cosine_similarities.append(m_data[str(M)]["cosine_similarity"]) 
        orthogonalities.append(m_data[str(M)]["orthogonality"])
        if "cosine_similarity_shifted" in m_data[str(M)]:
            cosine_similarities_shifted.append(m_data[str(M)]["cosine_similarity_shifted"])

    return {
        "min_M": data["min_M"],
        "max_M": data["max_M"],
        "ground_truth_modularity": data["ground_truth_modularity"],
        "modularities": modularities,
        "cosine_similarities": cosine_similarities,
        "cosine_similarities_shifted": cosine_similarities_shifted,
        "NMIs": NMIs,
        "orthogonality": orthogonalities
    }

def plot_results(img_name, mode, save_plots):
    annealer_data = load_data(img_name, mode, True)
    eigensolver_data = load_data(img_name, mode, False)
    
    annealer_data = preprocess_data(annealer_data)
    eigensolver_data = preprocess_data(eigensolver_data)

    if save_plots:
        path = get_results_path(img_name)
        maybe_create_output_dir(os.path.join(path, "plots"))

    prepare_plots(img_name, mode, annealer_data, eigensolver_data, save_plots)

    if not save_plots:
        plt.show()

if __name__ == "__main__":
    # img_name = "bubbles"
    # mode = Mode.LANCZOS

    # plot_results(img_name, mode, save_plots=True)

    for img_name in ["bubbles", "window", "maze", "desert"]:
        for mode in [Mode.LANCZOS, Mode.GRAM_SCHMIDT]:
            plot_results(img_name, mode, save_plots=True)