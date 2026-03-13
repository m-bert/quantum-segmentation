import os

import numpy as np
import networkx as nx

from utils.img_utils import load_image, create_image_from_graph, draw_image
from utils.graph_utils import convert_to_graph, visualize_graph
from utils.file_utils import get_imgs_names, maybe_create_output_dir

from Qommunity.samplers.regular.dqm_sampler import DQMSampler
from Qommunity.samplers.regular.louvain_sampler import LouvainSampler
from Qommunity.searchers.regular_searcher import RegularSearcher

from Qommunity.samplers.hierarchical.advantage_sampler import AdvantageSampler
from Qommunity.searchers.hierarchical_searcher import (
    HierarchicalSearcher,
)

# BETA = 100
BETA = 900

def process_image(img_name):
    maybe_create_output_dir(f"./q_out/{img_name}")
    img = load_image(f"./img/{img_name}.png")

    graph = convert_to_graph(img, BETA)

    # visualize_graph(graph, (img.shape[1], img.shape[0]), f"q_out/{img_name}/graph.png")

    # DQM
    dqm = DQMSampler(G=graph, time=60, cases=10, resolution=1, use_weights=True)
    dqm_searcher = RegularSearcher(dqm)
    communities = dqm_searcher.community_search()

    # Hierarchical
    # advantage = AdvantageSampler(G=graph, resolution=1, use_weights=True, use_clique_embedding=True)
    # searcher = HierarchicalSearcher(advantage)
    # communities = searcher.hierarchical_community_search()
    print(len(communities))
    print(communities)


    for community in communities:
        community_color = np.random.randint(0, 255, size=3)
        
        for node in community:
            graph.nodes[node]['color'] = community_color

    visualize_graph(graph, (img.shape[1], img.shape[0]), f"q_out/{img_name}/segmented_graph.png")

    new_img = create_image_from_graph(graph, (img.shape[1], img.shape[0]))

    draw_image(new_img, f"q_out/{img_name}/segmented.png")


if __name__ == "__main__":
    maybe_create_output_dir("./q_out")
    
    process_image("1")
    # process_image("6")
    # process_image("7")
    # process_image("big")



