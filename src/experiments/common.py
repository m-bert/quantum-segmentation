import os
import sys
from pathlib import Path

from enum import Enum

sys.path.append(str(Path(__file__).resolve().parent.parent))

class Mode(Enum):
    GRAM_SCHMIDT = "gram_schmidt"
    LANCZOS = "lanczos"

def get_results_path(img_name):
    return os.path.join(os.path.dirname(__file__), "results", img_name)

def get_image_path(img_name):
    return os.path.join(os.path.dirname(__file__), "..", "..", "img", "krylov_images", f"{img_name}.png")
