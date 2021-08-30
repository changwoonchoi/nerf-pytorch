import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


def colored_mask_to_label_map_np(colored_mask, color_list):
    """
    params:
        colored_mask: (H, W, 3)
        color_list: colors of labels
    returns:
        label: (H, W) stores instance num
    """
    f = lambda label, i: np.where(np.all(colored_mask == color_list[i], axis=-1), i, label)
    label_init = np.zeros(colored_mask.shape[:-1], dtype=np.int32)
    return reduce(f, list(range(len(color_list))), label_init)


def label_to_colored_label(label, label_color_list):
    colored_label = torch.zeros([*label.shape, 3], dtype=torch.uint8)
    for i in range(len(label_color_list)):
        mask_i = label == i
        colored_label[mask_i] = label_color_list[i].to(colored_label.device)

    return colored_label


class LabelEncoder:
    def __init__(self, label_color_list_np) -> None:
        self.label_color_list_np = label_color_list_np
        self.label_color_list = torch.from_numpy(label_color_list_np).to(torch.uint8)
        self.label_number = len(label_color_list_np)

    def get_dimension(self):
        pass

    def encode_np(self, label_np):
        pass

    def encode(self, label):
        pass

    def decode(self, encoded_label):
        pass

    def encoded_label_to_colored_label(self, encoded_label):
        label = self.decode(encoded_label)
        return label_to_colored_label(label, self.label_color_list)

    def error(self, output_encoded_label, target_label, **kwargs):
        target_encoded_label = self.encode(target_label)

        if len(target_encoded_label.shape) == 1:
            # output_encoded_label = output_encoded_label[:,0]
            target_encoded_label = torch.unsqueeze(target_encoded_label, 1)
        criterion = nn.MSELoss()
        return criterion(output_encoded_label, target_encoded_label)


class OneHotLabelEncoder(LabelEncoder):
    def encode(self, label):
        return F.one_hot(label, num_classes=self.label_number)

    def encode_np(self, label_np):
        return np.eye(self.label_number)[label_np]

    def decode(self, encoded_label):
        return torch.argmax(encoded_label, dim=-1)

    def error(self, output_encoded_label, target_label, **kwargs):
        data_bias = torch.tensor([torch.sum(target_label == k).item() for k in range(self.label_number)])
        if kwargs.get("fixed_CE_weight", False):
            CEloss = nn.CrossEntropyLoss()
        else:
            instance_weights = F.normalize(torch.ones(self.label_number) / data_bias, dim=0)
            CEloss = nn.CrossEntropyLoss(weight=instance_weights)
        return CEloss(output_encoded_label, target_label.long())

    def get_dimension(self):
        return self.label_number


class ScalarLabelEncoder(LabelEncoder):
    def encode(self, label):
        encoded_label = (label.float() + 0.5) / self.label_number
        return encoded_label

    def encode_np(self, label_np):
        return (label_np.astype(np.float32) + 0.5) / self.label_number

    def decode(self, encoded_label):
        index = torch.floor(encoded_label * self.label_number).long()
        index = torch.clip(index, min=0, max=self.label_number-1)
        index = torch.squeeze(index, -1)
        return index

    def get_dimension(self):
        return 1


class ColoredLabelEncoder(LabelEncoder):
    def encode(self, label):
        return self.label_color_list[label.long()].float().cuda() / 255.0

    def encode_np(self, label_np):
        return self.label_color_list_np[label_np].astype(np.float32) / 255.0

    def encoded_label_to_colored_label(self, encoded_label):
        return encoded_label * 255.0

    def get_dimension(self):
        return 3

    def decode(self, encoded_label):
        # this is useless
        return


def get_label_encoder(instance_color_list, label_encode_type) -> LabelEncoder:
    if label_encode_type == "one_hot":
        return OneHotLabelEncoder(instance_color_list)
    elif label_encode_type == "scalar":
        return ScalarLabelEncoder(instance_color_list)
    elif label_encode_type == "color":
        return ColoredLabelEncoder(instance_color_list)
