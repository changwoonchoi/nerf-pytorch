import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelEncoder:
    def __init__(self, label_color_list_np) -> None:
        self.label_color_list_np = label_color_list_np
        self.label_color_list = torch.from_numpy(label_color_list_np).to(torch.uint8)
        self.label_number = len(label_color_list_np)

    def get_dimension(self):
        pass

    def encode_np(self, label_np):
        pass

    def decode(self, encoded_label):
        pass

    def encoded_label_to_colored_label(self, encoded_label):
        label = self.decode(encoded_label)
        colored_label = torch.zeros([encoded_label.shape[0], encoded_label.shape[1], 3])
        for i in range(len(self.color_list)):
            mask_i = label == i
            colored_label[mask_i] = self.color_list[i].float().to(colored_label.device)

        return colored_label

    def error(self, output_encoded_label, target_encoded_label, target_label, **kwargs):
        return nn.MSELoss(output_encoded_label, target_encoded_label)


class OneHotLabelEncoder(LabelEncoder):
    def encode_np(self, label_np):
        return np.eye(self.label_number)[label_np]
    
    def decode(self, encoded_label):
        return torch.argmax(encoded_label, dim=-1)
    
    def error(self, output_encoded_label, target_encoded_label, target_label, **kwargs):
        data_bias = torch.tensor([torch.sum(target_label == k).item() for k in range(self.label_number)])
        if kwargs.get("fixed_CE_weight", False):
            bg_index = torch.argmax(data_bias).item()
            instance_weights = torch.ones(self.label_number)
            instance_weights[bg_index] /= 20
        else:
            instance_weights = F.normalize(torch.ones(self.label_number) / data_bias, dim=0)
        CEloss = nn.CrossEntropyLoss(weight=instance_weights)
        return CEloss(output_encoded_label, target_label.long())

    def get_dimension(self):
        return self.label_number

class ScalarLabelEncoder(LabelEncoder):
    def encode_np(self, label_np):
        return label_np / self.label_number

    def decode(self, encoded_label):
        return encoded_label * self.label_number

    def get_dimension(self):
        return 1

class ColoredLabelEncoder(LabelEncoder):
    def encode_np(self, label_np):
        return self.label_color_list_np[label_np]
    
    def encoded_label_to_colored_label(self, encoded_label):
        return encoded_label
    
    def get_dimension(self):
        return 3
