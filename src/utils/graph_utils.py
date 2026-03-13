import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools

from utils.img_utils import pixels_distance, find_max_distance

def visualize_graph(graph, size, path_to_save = None):
    rows, cols = size

    nodes_positions = {r * cols + c: (c, -r) for c in range(cols) for r in range(rows) }

    plt.figure(figsize=(9, 9))

    node_colors = [
        tuple([v / 255 for v in graph.nodes[n]['color']])
        for n in graph.nodes
    ] if 'color' in next(iter(graph.nodes(data=True)))[1] else 'lightblue'

    nx.draw(
        graph,
        pos=nodes_positions,
        node_color=node_colors,
        node_size=200,
        edge_color='gray',
        with_labels=True,
        font_size = 6
    )

    edge_labels = {edge: round(weight, 5) for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}

    nx.draw_networkx_edge_labels(
        graph,
        pos=nodes_positions,
        edge_labels=edge_labels,
        font_size=6
    )

    plt.axis("equal")

    if(path_to_save):
        plt.savefig(path_to_save)
        plt.close()
    else:
        plt.show()

    return 

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

    # Use itertools.combinations to generate unique node pairs
    for v, u in itertools.combinations(range(n), 2):
        x1, y1 = v % img.shape[1], v // img.shape[1]
        x2, y2 = u % img.shape[1], u // img.shape[1]

        weight = edge_weight(img[y1, x1], img[y2, x2], max_distance, beta)
        G.add_edge(v, u, weight=weight)

    return G