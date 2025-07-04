from __future__ import absolute_import
import os
import errno

from .osutils import mkdir_if_missing
import cv2


def separate_folder(file_name_path):
    directory, file_name = os.path.split(file_name_path)
    _, directory_name = os.path.split(directory)

    return directory_name, file_name
    

def save_feature_maps(heatmap_list, save_path="./feature_maps"):
    for key, value in heatmap_list.items():
        directory_name, file_name = separate_folder(key)
        save_dir = save_path + '/' + directory_name
        mkdir_if_missing(save_dir)
            
        print("Saved image at: ", save_dir + '/' + file_name)
        cv2.imwrite(save_dir  + '/' + file_name, value)
    return
