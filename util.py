import os

import face_recognition


def get_image_array(img_path):
    return face_recognition.load_image_file(img_path)


def create_directory(dir_name):
    os.makedirs(dir_name)
