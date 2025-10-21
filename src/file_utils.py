import os

def get_imgs_names(imgs_dir):
    imgs_paths = []

    for file_name in os.listdir(imgs_dir):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            imgs_paths.append(os.path.splitext(file_name)[0])

    return imgs_paths

def maybe_create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)