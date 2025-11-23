from img_utils import load_image
from graph_utils import convert_to_graph

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import differential_evolution

import networkx as nx

img_name = "big"
img = load_image(f"./img/{img_name}.png")

def ground_truth_communities():
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

def f(params):
    beta, resolution = params

    ground_truth = ground_truth_communities()
    
    graph = convert_to_graph(img, beta)
    communities = nx.community.louvain_communities(graph, resolution=resolution)

    diff = np.abs(len(ground_truth) - len(communities))

    for _ in range(diff):
        if len(communities) > len(ground_truth):
            ground_truth.append(set())
        elif len(communities) < len(ground_truth):
            communities.append(set())
    
    nmi = normalized_mutual_info_score(ground_truth, communities)

    # Penalize large differences in number of communities
    return -nmi + 0.1 * diff



result = differential_evolution(f, bounds=[(100, 1000),(0.5, 0.6)])
print("=== Optimal parameters ===")
print(f"Beta: {result.x[0]}")
print(f"Gamma: {result.x[1]}")