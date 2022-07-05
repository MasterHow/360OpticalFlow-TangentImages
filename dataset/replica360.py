import os
import os.path as osp
from glob import glob

from .base_flow import FlowDataset
from utils import augmentor


class Replica360(FlowDataset):

    def __init__(self,
                 aug_params=None,
                 split='//released',
                 root='datasets/Replica360',
                 dstype='circ',
                 valid=False,
                 back_flow=False):
        super(Replica360, self).__init__(aug_params)

        if valid:
            self.valid = True
            self.valid_list = []
            self.augmentor = augmentor.FlowValidAugmentor(**aug_params)

        map_list = self.folder_name(root + split + '//')

        # get dstype info
        dstype_list = [i.split("_")[-1] for i in map_list]

        for motion_type, map in zip(dstype_list, map_list):
            image_list = []
            if motion_type != dstype:
                pass
            elif motion_type == dstype:
                image_list += sorted(glob(osp.join(root + split + '//' + map, '*.jpg')))
                # zip image list
                for i in range(len(image_list) - 1):
                    self.image_list += [[image_list[i], image_list[i + 1]]]
                    self.extra_info += [i]  # frame_id
                self.flow_list += sorted(glob(osp.join(root + split + '//' + map, '*.flo')))[:-2]   # remove last flo
                if valid:
                    self.valid_list += sorted(glob(osp.join(root + split + '//' + map, '*.png')))[:-1]

        # remove backward flow
        if not back_flow:
            for flo_file in self.flow_list:
                if flo_file.split("_")[-2] == 'backward':
                    self.flow_list.remove(flo_file)

        pass

    def folder_name(self, file_dir):
        dir_list = []
        for root, dirs, files in os.walk(file_dir):
            dir_list.append(dirs)
        return dir_list[0]
