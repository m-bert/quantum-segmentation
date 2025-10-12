import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img, dtype=np.int16)

def draw_image(img, path_to_save = None):
    plt.imshow(img)
    plt.axis('off')

    if path_to_save:
        plt.savefig(path_to_save)
        plt.close()
    else:
        plt.show()

    return

def pixels_distance(p1, p2):
    return np.sum((p1 - p2) ** 2)

def find_max_distance(img):
    current_right = img[:, :-1]
    neighbor_right = img[:, 1:]

    current_down = img[:-1, :]
    neighbor_down = img[1:, :]

    d_vec = np.vectorize(pixels_distance, signature='(n),(n)->()')

    right_distances = d_vec(current_right, neighbor_right)
    down_distances  = d_vec(current_down, neighbor_down)

    return max(right_distances.max(), down_distances.max())

def create_image_from_graph(graph, size):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    for (x, y), data in graph.nodes(data=True):
        img[y][x] = data['color'] if 'color' in data else (255, 255, 255)

    for v in graph.nodes:
        img[v[1]][v[0]] = graph.nodes[v]['color']

    return img