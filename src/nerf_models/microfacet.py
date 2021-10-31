import torch


def fresnel_schlick_roughness(cosTheta, F0, roughness):
	cosTheta = cosTheta[..., None]
	roughness = roughness[..., None]
	F1 = torch.maximum(1.0 - roughness, F0) - F0
	return F0 + F1 * torch.pow(torch.clip(1.0 - cosTheta, 0.0, 1.0), 5.0)