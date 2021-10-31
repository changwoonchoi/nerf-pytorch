import torch
from typing import *
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def histogram(img: torch.Tensor, channels: List[int]=[32, 32, 32]) -> torch.Tensor:
    """
    Returns a color histogram of an input image

    Args:
        img: (H, W, 3)
        mask: (H, W)
        channels: List of length 3 containing number of bins per each channel
    Returns:
        hist: Histogram of shape (*channels)
    """
    tgt_img = img.clone().detach()
    max_rgb = torch.LongTensor([255] * 3).to(tgt_img.device)
    bin_size = torch.ceil(max_rgb.float() / torch.tensor(channels).float().to(tgt_img.device)).long()

    # when RGB value is in [0, 1]
    if tgt_img.max() <= 1:
        tgt_img = (tgt_img * max_rgb.reshape(-1, 3)).long()

    if len(img.shape) == 3:
        tgt_rgb = tgt_img.reshape(-1, 3).long()
        tgt_rgb = tgt_rgb // bin_size.reshape(-1, 3)
        tgt_rgb = tgt_rgb[:, 0] + channels[0] * tgt_rgb[:, 1] + channels[0] * channels[1] * tgt_rgb[:, 2]

        hist = torch.bincount(tgt_rgb, minlength=channels[0] * channels[1] * channels[2]).float()
        hist = hist.reshape(*channels)
    else:  # Batched input
        tgt_img = tgt_img // bin_size.reshape(-1, 3)
        tgt_img = tgt_img[..., 0] + channels[0] * tgt_img[..., 1] + channels[0] * channels[1] * tgt_img[..., 2]  # (B, H, W)
        tgt_img = tgt_img.reshape(tgt_img.shape[0], -1).long()  # (B, H * W)
        hist = torch.zeros([tgt_img.shape[0], channels[0] * channels[1] * channels[2]], device=tgt_img.device, dtype=torch.long).scatter_add(
            dim=-1, index=tgt_img, src=torch.ones_like(tgt_img, dtype=torch.long))
        hist = hist.reshape([hist.shape[0], *channels])

    return hist


def merge_cluster(centers, weights, th, minimum=4):
    """
    merge clusters closer than threshold
    """
    coords = centers.copy()
    new_label_map = np.arange(len(centers))
    left_ids = [i for i in range(len(centers))]

    def get_parent_id(label_map, idx):
        while True:
            idx = label_map[idx]
            if idx == label_map[idx]:
                return idx

    counts = weights.copy()
    while len(left_ids) > minimum:
        # make distance matrix
        dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        np.fill_diagonal(dist_matrix, 1e6)

        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        assert min_idx[0] < min_idx[1], "min_idx must in upper triangle matrix"

        min_val = dist_matrix[min_idx]
        if min_val > th:
            break
        else:
            # merge centroids
            new_count = counts[min_idx[0]] + counts[min_idx[1]]
            new_center = (coords[min_idx[0]] * counts[min_idx[0]] + coords[min_idx[1]] * counts[min_idx[1]]) / new_count
            coords = np.delete(coords, [min_idx[1]], axis=0)
            counts = np.delete(counts, [min_idx[1]])
            coords[min_idx[0]] = new_center
            counts[min_idx[0]] = new_count

            original_idx0 = left_ids[min_idx[0]]
            original_idx1 = left_ids[min_idx[1]]

            parent0 = get_parent_id(new_label_map, original_idx0)
            parent1 = get_parent_id(new_label_map, original_idx1)

            parent_id = min(parent0, parent1)
            for i in range(len(new_label_map)):
                parent_i = get_parent_id(new_label_map, i)
                if parent_i == parent0 or parent_i == parent1:
                    new_label_map[i] = parent_id

            del left_ids[min_idx[1]]

    new_indices = [i for i in range(len(centers))]
    new_indices = np.asarray(new_indices)

    for i in range(len(new_label_map)):
        parent_id_i = get_parent_id(new_label_map, i)
        for new_index, parent_id in enumerate(left_ids):
            if parent_id_i == parent_id:
                new_indices[i] = new_index
    return coords, new_indices, counts


def get_basecolor(img, use_hist=False, n_clusters=8, cluster_th=0.1, n_clusters_minimum=4, visualize=False, normalize=True):
    if normalize:
        chrom = torch.Tensor(img).float() / (torch.linalg.norm(torch.Tensor(img).float(), dim=-1, keepdim=True) + 1e-10)
    else:
        chrom = torch.Tensor(img).float()
    if not use_hist:
        cluster = MiniBatchKMeans(n_clusters=n_clusters).fit(chrom.cpu().numpy().reshape(-1, 3))
    else:
        # make histogram
        hist = histogram(chrom, channels=[32, 32, 32])
        if len(chrom.shape) > 3:
            hist = torch.sum(hist, dim=0)
        coords = torch.nonzero(hist)
        weights = [hist[coords[i][0], coords[i][1], coords[i][2]] for i in range(coords.shape[0])]
        weights = torch.stack(weights, dim=0)
        # weighted k-means clustering
        cluster = KMeans(n_clusters=n_clusters).fit(coords.cpu().numpy(), weights.cpu().numpy())

    init_centers = cluster.cluster_centers_  # (n_cluster, 3)
    labels = cluster.labels_  # (n_cluster, )
    #print("Cluster centers (before merge)", cluster.cluster_centers_)

    centroid_weights = np.array([len(labels[labels == i]) for i in range(n_clusters)])
    new_centroid, new_indices, _ = merge_cluster(init_centers, centroid_weights, th=cluster_th, minimum=n_clusters_minimum)
    #print("Cluster centers (after merge)", new_centroid)
    #print("New label map", new_indices)
    if visualize:
        import matplotlib.pyplot as plt
        predicted_label = labels.reshape((img.shape[0], img.shape[1], img.shape[2]))
        predicted_image = cluster.cluster_centers_[labels].reshape(img.shape)
        merged_label = new_indices[predicted_label]
        merged_image = new_centroid[merged_label]

        for i in range(4):
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(predicted_label[i])
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(merged_label[i])
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(predicted_image[i])
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(merged_image[i])
        plt.show()

    return new_centroid
