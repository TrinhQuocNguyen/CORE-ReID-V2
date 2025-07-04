from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

import os.path as osp

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
import yaml

# Load the global config
with open("global_config.yaml", "r") as file:
    global_config = yaml.safe_load(file)

class VeRiWild(BaseImageDataset):
    """VeRi-Wild.

    Reference:
        Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.

    URL: `<https://github.com/PKU-IMRE/VERI-Wild>`_

    Train dataset statistics:
        - identities: 30671.
        - images: 277797.
    """
    # dataset_dir = "VeRI-Wild"
    dataset_name = "veriwild"
    dataset_dir = global_config["data_path"] + 'VeRI-Wild'
    

    def __init__(self, root, query_list='', gallery_list='', verbose=True, **kwargs):
        super(VeRiWild, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.vehicle_info = osp.join(self.dataset_dir, 'train_test_split/vehicle_info.txt')
        
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list_start0.txt')
        
        if query_list and gallery_list:
            self.query_list = query_list
            self.gallery_list = gallery_list
        else:
            self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_10000_id_query.txt')
            self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_10000_id.txt')

        self._check_before_run()

        self.imgid2vid, self.imgid2camid, self.imgid2imgpath = self.process_vehicle(self.vehicle_info)
        

        train = self._process_dir(self.train_list)
        query = self._process_dir(self.query_list, relabel=False)
        gallery = self._process_dir(self.gallery_list, relabel=False)
        

        self.train = train
        self.query = query
        self.gallery = gallery
        
        if verbose:
            print("=> VeRI-Wild loaded")
            self.print_dataset_statistics(train, query, gallery)
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.image_dir):
            raise RuntimeError("'{}' is not available".format(self.image_dir))
        if not osp.exists(self.vehicle_info):
            raise RuntimeError("'{}' is not available".format(self.vehicle_info))
        if not osp.exists(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))
        if not osp.exists(self.query_list):
            raise RuntimeError("'{}' is not available".format(self.query_list))
        if not osp.exists(self.gallery_list):
            raise RuntimeError("'{}' is not available".format(self.gallery_list))
        
    def _process_dir(self, img_list, relabel=False):
        img_list_lines = open(img_list, 'r').readlines()
        # print(self.imgid2camid)

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = int(line.split('/')[0])
            imgid = line.split('/')[1].split('.')[0]
            camid = int(self.imgid2camid[imgid])
            if relabel:
                vid = f"{self.dataset_name}_{vid}"
                camid = f"{self.dataset_name}_{camid}"
            dataset.append((self.imgid2imgpath[imgid], vid, camid))

        assert len(dataset) == len(img_list_lines)
        return dataset

    def process_vehicle(self, vehicle_info):
        imgid2vid = {}
        imgid2camid = {}
        imgid2imgpath = {}
        vehicle_info_lines = open(vehicle_info, 'r').readlines()

        for idx, line in enumerate(vehicle_info_lines[0:]):
            vid = line.strip().split('/')[0]
            imgid = line.strip().split(';')[0].split('/')[1]
            camid = line.strip().split(';')[1]
            img_path = osp.join(self.image_dir, vid, imgid + '.jpg')
            imgid2vid[imgid] = vid
            imgid2camid[imgid] = camid
            imgid2imgpath[imgid] = img_path
        print("len(imgid2vid): ", len(imgid2vid))
        print("len(vehicle_info_lines): ", len(vehicle_info_lines))
        assert len(imgid2vid) == len(vehicle_info_lines)
        return imgid2vid, imgid2camid, imgid2imgpath
