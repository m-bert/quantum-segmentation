import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return np.array(img, dtype=np.int64)

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
    max_distance = -1

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if x > 0:
                dist = pixels_distance(img[y][x], img[y][x - 1])
                if dist > max_distance:
                    max_distance = dist

            if y > 0:
                dist = pixels_distance(img[y][x], img[y - 1][x])
                if dist > max_distance:
                    max_distance = dist

            if x > 0 and y > 0:
                dist = pixels_distance(img[y][x], img[y - 1][x - 1])
                if dist > max_distance:
                    max_distance = dist

            if x < img.shape[1] - 1 and y > 0:
                dist = pixels_distance(img[y][x], img[y - 1][x + 1])
                if dist > max_distance:
                    max_distance = dist

    return max_distance

def create_image_from_graph(graph, size):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    for v in graph.nodes:
        img[v // size[1]][v % size[1]] = graph.nodes[v]['color']

    return img