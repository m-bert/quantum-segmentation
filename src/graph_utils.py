import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from img_utils import pixels_distance, find_max_distance

def visualize_graph(graph, size, path_to_save = None):
    rows, cols = size

    nodes_positions = {(c, r): (c, -r) for r in range(rows) for c in range(cols)}

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

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            G.add_node((x, y), color=tuple(img[y, x]))

            if x > 0:
                weight = edge_weight(img[y, x], img[y, x-1], max_distance, beta)
                G.add_edge((x, y), (x-1, y), weight=weight)

            if y > 0:
                weight = edge_weight(img[y, x], img[y-1, x], max_distance, beta)
                G.add_edge((x, y), (x, y-1), weight=weight)

            if x > 0 and y > 0:
                weight = edge_weight(img[y, x], img[y-1, x-1], max_distance, beta)
                G.add_edge((x, y), (x-1, y-1), weight=weight)

            if x < img.shape[0] - 1 and y > 0:
                weight = edge_weight(img[y, x], img[y-1, x+1], max_distance, beta)
                G.add_edge((x, y), (x+1, y-1), weight=weight)

    return G

def convert_communities_to_graph(communities, img, beta):
    G = nx.Graph()
    max_distance = find_max_distance(img)


    for i, community in enumerate(communities):
        first_node = next(iter(community))
        G.add_node(i, color=img[first_node[1], first_node[0]])

    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            weight = edge_weight(G.nodes[i]['color'], G.nodes[j]['color'], max_distance, beta)
            G.add_edge(i, j, weight=weight)


    # node_colors = [
    #     tuple([v / 255 for v in G.nodes[n]['color']])
    #     for n in G.nodes
    # ] if 'color' in next(iter(G.nodes(data=True)))[1] else 'lightblue'

    # nx.draw(
    #     G,
    #     node_color=node_colors,
    #     node_size=200,
    #     edge_color='gray',
    #     with_labels=True,
    #     font_size = 6
    # )

    # edge_labels = {edge: round(weight, 5) for edge, weight in nx.get_edge_attributes(G, 'weight').items()}

    # nx.draw_networkx_edge_labels(
    #     G,
    #     pos=nx.spring_layout(G),
    #     edge_labels=edge_labels,
    #     font_size=6
    # )

    # plt.axis("equal")

    # plt.show()

    return G