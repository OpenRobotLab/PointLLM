import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from pointllm.utils import *
from pointllm.data.utils import *

class ModelNet(Dataset):
    def __init__(self, config_path, split, subset_nums=-1, use_color=False):
        """
        Args:
            data_args:
                split: train or test
        """
        super(ModelNet, self).__init__()

        if config_path is None:
            # * use the default config file in the same dir
            config_path = os.path.join(os.path.dirname(__file__), "modelnet_config", "ModelNet40.yaml")

        config = cfg_from_yaml_file(config_path)
        # * check data path
        self.root = config["DATA_PATH"]
    
        if not os.path.exists(self.root):
            print(f"Data path {self.root} does not exist. Please check your data path.")
            exit()

        self.npoints = config.npoints
        self.num_category = config.NUM_CATEGORY # * should be 40
        self.random_sample = config.random_sampling
        self.use_height = config.use_height
        self.use_normals = config.USE_NORMALS
        self.subset_nums = subset_nums
        self.normalize_pc = True
        self.use_color = use_color

        if self.use_height or self.use_normals:
            print(f"Warning: Usually we don't use height or normals for shapenet but use_height: {self.use_height} and \
                  use_normals: {self.use_normals}.")

        self.split = split
        assert (self.split == 'train' or self.split == 'test')

        self.catfile = os.path.join(os.path.dirname(__file__), "modelnet_config", 'modelnet40_shape_names_modified.txt')

        # "tv_stand" -> "tv stand"
        self.categories = [line.rstrip() for line in open(self.catfile)] # * list of category names

        self.save_path = os.path.join(self.root,
                                    'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, self.split, self.npoints))

        print('Load processed data from %s...' % self.save_path)
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f) # * ndarray of N, C: (8192, 6) (xyz and normals)
        
        if self.subset_nums > 0:
            # * set random seed
            import random
            random.seed(0)
            # * random choose subset_nums
            idxs = random.sample(range(len(self.list_of_labels)), self.subset_nums)
            self.list_of_labels = [self.list_of_labels[idx] for idx in idxs]
            self.list_of_points = [self.list_of_points[idx] for idx in idxs]

        # * print len
        print(f"Load {len(self.list_of_points)} data from {self.save_path}.")

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        if  self.npoints < point_set.shape[0]:
            if self.random_sample:
                # * random sample 
                point_set = point_set[np.random.choice(point_set.shape[0], self.npoints, replace=False)]
            else:
                point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)
        
        point_set = np.concatenate((point_set, np.zeros_like(point_set)), axis=-1) if self.use_color else point_set

        return point_set, label.item() # * ndarray, int
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.split == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()

        if self.normalize_pc:
            # * modelnet point cloud is already normalized
            current_points = self.pc_norm(current_points)

        current_points = torch.from_numpy(current_points).float() # * N, C tensors
        label_name = self.categories[int(label)]

        data_dict = {
            "indice": index, # * int
            "point_clouds": current_points, # * tensor of N, C
            "labels": label, # * int
            "label_names": label_name # * str
        }

        return data_dict
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ModelNet Dataset')

    parser.add_argument("--config_path", type=str, default=None, help="config file path.")
    parser.add_argument("--split", type=str, default="test", help="train or test.")
    parser.add_argument("--subset_nums", type=int, default=200)

    args = parser.parse_args()

    dataset = ModelNet(config_path=args.config_path, split=args.split, subset_nums=args.subset_nums)

    # * get the first item
    print(dataset[0])