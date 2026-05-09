import matplotlib.pyplot as plt
import os
import json

from common import Mode

def prepare_plots(annealer_data, eigensolver_data):
    annealer_color = 'royalblue'
    eigensolver_color = 'limegreen'
    eigensolver_shifted_color = 'forestgreen'
    ground_truth_color = 'red'

    min_m = annealer_data["min_M"]
    max_m = annealer_data["max_M"]

    fig, ax = plt.subplots(1, 4, figsize=(16, 5))

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

    ax[1].plot(range(min_m, max_m + 1), annealer_data["cosine_similarities"], '-o', markersize=3, label='Annealer', color=annealer_color)
    ax[1].plot(range(min_m, max_m + 1), eigensolver_data["cosine_similarities"], '-o', markersize=3, label='Eigensolver', color=eigensolver_color)
    ax[1].plot(range(min_m, max_m + 1), eigensolver_data["cosine_similarities_shifted"], '-o', markersize=3, label='Eigensolver Shifted', color=eigensolver_shifted_color)
    ax[1].set_xlabel("M")
    ax[1].set_xticks(range(min_m, max_m + 1))
    ax[1].set_ylabel("Cosine similarity with leading eigenvector")
    ax[1].spines[["top", "right"]].set_visible(False)
    ax[1].set_title("Cosine similarity vs Krylov subspace dimension")
    ax[1].legend(loc='lower right')
    ax[1].grid(True)

    ax[2].plot(range(min_m, max_m + 1), annealer_data["NMIs"], '-o', markersize=3, label='Annealer', color=annealer_color)
    ax[2].plot(range(min_m, max_m + 1), eigensolver_data["NMIs"], '-o', markersize=3, label='Eigensolver', color=eigensolver_color)
    ax[2].set_xlabel("M")
    ax[2].set_xticks(range(min_m, max_m + 1))
    ax[2].set_ylabel("Normalized Mutual Information (NMI)")
    ax[2].spines[["top", "right"]].set_visible(False)
    ax[2].set_title("NMI vs Krylov subspace dimension")
    ax[2].legend(loc='lower right')
    ax[2].grid(True)

    ax[3].plot(range(min_m, max_m + 1), annealer_data["orthogonality"], '-o', markersize=3, label='Annealer', color=annealer_color)
    ax[3].plot(range(min_m, max_m + 1), eigensolver_data["orthogonality"], '-o', markersize=3, label='Eigensolver', color=eigensolver_color)
    ax[3].plot(range(min_m, max_m + 1), range(min_m, max_m + 1), '--', color=ground_truth_color, label='Ideal Orthogonality')
    ax[3].set_xlabel("M")
    ax[3].set_xticks(range(min_m, max_m + 1))
    ax[3].set_ylabel("Orthogonality")
    ax[3].spines[["top", "right"]].set_visible(False)
    ax[3].set_title("Orthogonality vs Krylov subspace dimension")
    ax[3].legend(loc='lower right')
    ax[3].grid(True)

    fig.tight_layout()
    return fig, ax

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

def plot_results(img_name, mode):
    annealer_data = load_data(img_name, mode, True)
    eigensolver_data = load_data(img_name, mode, False)
    
    annealer_data = preprocess_data(annealer_data)
    eigensolver_data = preprocess_data(eigensolver_data)

    prepare_plots(annealer_data, eigensolver_data)

    plt.show()

if __name__ == "__main__":
    plot_results("bubbles", Mode.LANCZOS)