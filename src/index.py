import os

import sys
from pathlib import Path

import numpy as np
import networkx as nx

from utils.img_utils import load_image, create_image_from_graph, draw_image
from utils.graph_utils import convert_to_graph, visualize_graph
from utils.file_utils import get_imgs_names, maybe_create_output_dir

sys.path.append(str(Path(__file__).resolve().parent))

BETA = 472
GAMMA = 1
# BETA = 100

def process_image(img_name):
    maybe_create_output_dir(f"./out/{img_name}")
    img = load_image(f"./img/{img_name}.png")

    graph = convert_to_graph(img, BETA)

    # visualize_graph(graph, img.shape[:2], f"out/{img_name}/graph.png")

    communities = nx.community.louvain_communities(graph, resolution=GAMMA)
    # print(len(communities))

    for community in communities:
        community_color = np.random.randint(0, 255, size=3)
        
        for node in community:
            graph.nodes[node]['color'] = community_color

    # visualize_graph(graph, img.shape[:2], f"out/{img_name}/segmented_graph.png")

    new_img = create_image_from_graph(graph, img.shape[:2])

    draw_image(new_img, f"out/{img_name}/segmented.png")


if __name__ == "__main__":
    maybe_create_output_dir("./out")
    
    process_image("real/brain_128")
    # imgs_names = get_imgs_names("./img")

    # for img_name in imgs_names:
    #     process_image(img_name)


