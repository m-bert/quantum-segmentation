import numpy as np
import networkx as nx

from img_utils import load_image, create_image_from_graph, draw_image
from graph_utils import convert_to_graph, visualize_graph

IMG_NAME = "1"
IMG_PATH = f"img/{IMG_NAME}.png"

BETA = 50

if __name__ == "__main__":
    img = load_image(IMG_PATH)
    graph = convert_to_graph(img, BETA)

    visualize_graph(graph, (img.shape[1], img.shape[0]), f"out/{IMG_NAME}_graph.png")

    communities = nx.community.louvain_communities(graph)

    for community in communities:
        community_color = np.random.randint(0, 255, size=3)
        
        for node in community:
            graph.nodes[node]['color'] = community_color

    visualize_graph(graph, (img.shape[1], img.shape[0]), f"out/{IMG_NAME}_segmented_graph.png")

    new_img = create_image_from_graph(graph, (img.shape[1], img.shape[0]))

    draw_image(new_img, f"out/{IMG_NAME}_segmented.png")

