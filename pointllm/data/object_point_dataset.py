import os
import json
import torch
import numpy as np

import copy
import transformers
from torch.utils.data import Dataset

from .utils import *


def make_object_point_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for Joint3Ddataset with text and point cloud data."""
    """Initialize datasets."""

    data_collator = DataCollatorForPointTextDataset(tokenizer=tokenizer)
    if data_args.split_train_val:
        print("Loading training datasets.")
        train_dataset = ObjectPointCloudDataset(
            split='train',
            data_path=data_args.data_path,
            anno_path=data_args.anno_path,
            pointnum=data_args.pointnum,
            conversation_types=data_args.conversation_types,
            tokenizer=tokenizer,
            use_color=data_args.use_color,
            data_args=data_args
        )
        print("Done!")
        if data_args.data_debug_num > 0:
            print('Debug mode, using training set as val set.')
            val_dataset = train_dataset
        else:
            # * make a val dataset
            print("Loading validation datasets.")
            val_dataset = ObjectPointCloudDataset(
                split='val', # * load train split
                data_path=data_args.data_path,
                anno_path=data_args.anno_path,
                pointnum=data_args.pointnum,
                conversation_types=data_args.conversation_types,
                tokenizer=tokenizer,
                use_color=data_args.use_color,
                data_args=data_args
            )
        return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
    else:
        # * use all data as training data
        train_dataset = ObjectPointCloudDataset(
            split='train',
            data_path=data_args.data_path,
            anno_path=data_args.anno_path,
            pointnum=data_args.pointnum,
            conversation_types=data_args.conversation_types,
            use_color=data_args.use_color,
            tokenizer=tokenizer,
            data_args=data_args
        )
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class ObjectPointCloudDataset(Dataset):
    """Dataset utilities for objaverse."""
    def __init__(self,
                 data_path=None,
                 anno_path=None,
                 tokenizer=None,
                 pointnum=8192,
                 split='train',
                 conversation_types=None, # * default is simple_des, used for stage1 pre-train
                 use_color=True,
                 data_args=None):

        """
        split: only considered when data_args.split_train_val is True.
        conversation_types: tuple, used to filter the data, default is ('simple_description'), other types is:
            "detailed_description", "single_round", "multi_round".
        tokenizer: load point clouds only if None
        """
        super(ObjectPointCloudDataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.data_path = data_path
        self.anno_path = anno_path
        self.tokenizer = tokenizer
        self.split = split 
        if conversation_types is None:
            self.conversation_types = ("simple_description",)
        else:
            self.conversation_types = conversation_types

        self.data_args = data_args
        self.normalize_pc = True
        self.use_color = use_color

        self.pointnum = pointnum
        self.point_backbone_config = data_args.point_backbone_config if data_args is not None else None
        self.point_indicator = '<point>'

        # Load the data list from JSON
        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)
        
        # * print the conversations_type
        print(f"Using conversation_type: {self.conversation_types}") 
        # * print before filtering
        print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")

        # * iterate the list and filter
        # * these two ids have corrupted colored point files, so filter them when use_color is True
        filter_ids = ['6760e543e1d645d5aaacd3803bcae524', 'b91c0711149d460a8004f9c06d3b7f38'] if self.use_color else []

        # Iterate the list, filter those "conversation_type" not in self.conversation_types
        self.list_data_dict = [
            data for data in self.list_data_dict 
            if data.get('conversation_type', 'simple_description') in self.conversation_types 
            and data.get('object_id') not in filter_ids
        ]

        # * print after filtering
        print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")
        # * print the size of different conversation_type
        for conversation_type in self.conversation_types:
            print(f"Number of {conversation_type}: {len([data for data in self.list_data_dict if data.get('conversation_type', 'simple_description') == conversation_type])}")

        if self.data_args is not None and self.data_args.data_debug_num > 0:
            self.list_data_dict = self.list_data_dict[:self.data_args.data_debug_num]
            # * print all the scan_id in debug mode, not using for loop
            print('Debug mode, using: ' + ' '.join([data['object_id'] for data in self.list_data_dict]))
        elif self.data_args is not None and self.data_args.split_train_val:
            # * split train and val with 9:1 ratios
            if self.split == 'train':
                self.list_data_dict = self.list_data_dict[:int(self.data_args.split_ratio * len(self.list_data_dict))]
                print(f"Train set size: {len(self.list_data_dict)}")
            else:
                self.list_data_dict = self.list_data_dict[int(self.data_args.split_ratio * len(self.list_data_dict)):]
                print(f"Val set size: {len(self.list_data_dict)}")

    def _load_point_cloud(self, object_id, type='objaverse'):
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id) 

    def _load_objaverse_point_cloud(self, object_id):
        filename = f"{object_id}_{self.pointnum}.npy"
        point_cloud = np.load(os.path.join(self.data_path, filename))

        if not self.use_color:
            point_cloud = point_cloud[:, :3]

        return point_cloud

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
        sources = self.list_data_dict[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"
        if self.point_indicator in sources[0]['conversations'][0]['value']:

            object_id = self.list_data_dict[index]['object_id']

            # Point cloud representation
            point_cloud = self._load_point_cloud(object_id) # * N, C
            if self.normalize_pc:
                point_cloud = self.pc_norm(point_cloud) # * need to norm since point encoder is norm

            if self.tokenizer is None:
                data_dict = dict(
                    point_clouds=torch.from_numpy(point_cloud.astype(np.float32)),
                    object_ids=object_id
                )
                return data_dict

            sources = preprocess_multimodal_point_cloud(
                copy.deepcopy([e["conversations"] for e in sources]), self.point_backbone_config, point_indicator=self.point_indicator)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess_v1(
            sources,
            self.tokenizer)

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # point exist in the data
        if self.point_indicator in self.list_data_dict[index]['conversations'][0]['value']:
            data_dict['point_clouds'] = torch.from_numpy(point_cloud.astype(np.float32))

        return data_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/objaverse_data", type=str,
                        help="Path to the data directory.")
    parser.add_argument("--anno_path", default=None, type=str, required=True,
                        help="Path to the annotation file.")
    parser.add_argument("--split", default='train', type=str, 
                        help="Whether to use the train or validation dataset.")
    parser.add_argument("--pointnum", default=8192, type=int,
                        help="Number of points in the point cloud.")
    parser.add_argument("--data_debug_num", default=0, type=int,
                        help="Number of data to debug with.")
    parser.add_argument("--split_train_val", default=False, type=bool,
                        help="Whether to split the dataset into training and validation.")
    parser.add_argument("--split_ratio", default=0.9, type=float,
                        help="The ratio of training to validation data.")
    parser.add_argument("--tokenizer_path", default=None, type=str, required=True,
                        help="Path to the tokenizer config file.")
    
    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    args.point_backbone_config = None

    # Initialize dataset
    dataset = ObjectPointCloudDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        pointnum=args.pointnum,
        split=args.split,
        tokenizer=tokenizer,
        data_args=args
    )

    # Example usage
    print(f'Dataset length: {len(dataset)}')

