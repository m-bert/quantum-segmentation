import os

import numpy as np
import networkx as nx

from img_utils import load_image, create_image_from_graph, draw_image
from graph_utils import convert_to_graph, visualize_graph, convert_communities_to_graph
from file_utils import get_imgs_names, maybe_create_output_dir

BETA = 100

def process_image(img_name):
    maybe_create_output_dir(f"./out/{img_name}")
    img = load_image(f"./img/{img_name}.png")

    graph = convert_to_graph(img, BETA)

    # visualize_graph(graph, (img.shape[1], img.shape[0]), f"out/{img_name}/graph.png")

    communities = nx.community.louvain_communities(graph, resolution=0.05)

    communities_graph = convert_communities_to_graph(communities, img, BETA)

    new_communities = nx.community.louvain_communities(communities_graph, resolution=1)


    for community in new_communities:
        community_color = np.random.randint(0, 255, size=3)
        
        for node in community:
            for original_node in communities[node]:
                graph.nodes[original_node]['color'] = community_color



    # visualize_graph(graph, (img.shape[1], img.shape[0]), f"out/{img_name}/segmented_graph.png")

    new_img = create_image_from_graph(graph, (img.shape[1], img.shape[0]))

    draw_image(new_img, f"out/{img_name}/segmented.png")


if __name__ == "__main__":
    maybe_create_output_dir("./out")
    
    process_image("6")
    # imgs_names = get_imgs_names("./img")

    # for img_name in imgs_names:
    #     process_image(img_name)


