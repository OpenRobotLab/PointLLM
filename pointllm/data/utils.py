from collections import OrderedDict, defaultdict

import transformers
from pointllm import conversation as conversation_lib
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import torch

import numpy as np
import os

IGNORE_INDEX = -100

# * Sample Usage:
# * from utils import LRUCache
# * cache = LRUCache(capacity, max_access_count)
# if self.cache is None:
#     info_data = self.multiview_scannet[info_index]
# else:
#     info_data = self.cache.get(info_index)
#     if info_data is None or self.cache.get_access_count(info_index) >= self.cache.max_access_count:
#         # If not in cache, or accessed max_access_count times, load it and put it in cache
#         info_data = self.multiview_scannet[info_index]
#         self.cache.put(info_index, info_data)
#         self.cache.reset_access_count(info_index)

class LRUCache:
    def __init__(self, capacity, max_access_count):
        self.cache = OrderedDict()
        self.access_count = defaultdict(int)
        self.capacity = capacity
        self.max_access_count = max_access_count

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value  # Put key as the newest one
        self.access_count[key] += 1
        return value

    def put(self, key, value):
        if key in self.cache:  # Update the value and put it as newest
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:  # If cache is full
            oldest_key = next(iter(self.cache))
            self.cache.popitem(last=False)  # Remove oldest item
            del self.access_count[oldest_key]  # Remove the corresponding access count
        self.cache[key] = value
        self.access_count[key] = 1

    def get_access_count(self, key):
        return self.access_count.get(key, 0)

    def reset_access_count(self, key):
        self.access_count[key] = 0


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2: # * can handle padded tokens
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: # * unk tokens in the dialogue will cause this.
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_multimodal_point_cloud(
    sources: Sequence[str],
    point_backbone_config: dict,
    point_indicator: str = "<point>",
) -> Dict:
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']

    for source in sources:
        for sentence in source:
            replace_token = default_point_patch_token * point_token_len 
            if point_backbone_config['mm_use_point_start_end']:
                replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)

    return sources

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

def load_ulip2_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=False):
    if not use_color:
        filename = f"{object_id}/{object_id}_{pointnum}.npz"
        point_cloud = np.load(os.path.join(data_path, filename))['arr_0'] # * pointnum, 3 array
    else:
        filename = f"{object_id}_{pointnum}.npy"
        point_cloud = np.load(os.path.join(data_path, filename))

    # * normalize
    point_cloud = pc_norm(point_cloud)

    return point_cloud

@dataclass
class DataCollatorForPointTextDataset(object):
    """Collate examples for mixed dataset with text and point cloud data."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'point_clouds' in instances[0]:
            point_clouds = [instance['point_clouds'] for instance in instances]
            if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds): # * point_clouds have different shapes
                batch['point_clouds'] = torch.stack(point_clouds)
            else:
                batch['point_clouds'] = point_clouds # * return as lists

        return batch

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_normalize(pc):
    """
    pc: Nx3 array
    This functions normalizes a point cloud to fit within a unit sphere.
    It first calculates the centroid of the point cloud and then subtracts
    it from all points before scaling all points to fit within a unit sphere.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc