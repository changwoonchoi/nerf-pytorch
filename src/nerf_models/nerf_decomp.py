import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.logging_utils
from nerf_models.positional_embedder import get_embedder
import os


# Model
class NeRFDecomp(nn.Module):
    def __init__(
            self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], instance_label_dimension=0,
            use_instance_feature_layer=False
    ):
        """
        NeRFDecomp Model
        params:
            D: Network Depth
            W: MLP dim
            input_ch: dim of x (position) (3 for R^3)
            input_ch_views: dim of d (direction) (3 for R^3 vector)
            output_ch:
            skips: list of layer numbers that are concatenated with x (position)
            instance_label_dimension: dimension of instance_label
            use_instance_feature_layer: use additional layer following Shuaifeng Zhi et al.
        """
        super(NeRFDecomp, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.skips = skips
        self.instance_label_dimension = instance_label_dimension
        self.use_instance_feature_layer = use_instance_feature_layer

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        self.feature_linear = nn.Linear(W, W)
        self.sigma_linear = nn.Linear(W, 1)
        if self.instance_label_dimension > 0:
            if self.use_instance_feature_layer:
                self.instance_feature_linear = nn.Linear(W, W // 2)
                self.instance_linear = nn.Linear(W // 2, self.instance_label_dimension)
            else:
                self.instance_feature_linear = None
                self.instance_linear = nn.Linear(W, self.instance_label_dimension)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def __str__(self):
        logs = ["[NeRFDecomp"]
        logs += ["\t- depth : {}".format(self.D)]
        logs += ["\t- width : {}".format(self.W)]
        logs += ["\t- input_ch : {}".format(self.input_ch)]
        logs += ["\t- output_ch : {}".format(self.output_ch)]
        logs += ["\t- instance_label_dimension : {}".format(self.instance_label_dimension)]
        logs += ["\t- use_instance_feature_layer : {}".format(self.use_instance_feature_layer)]
        return "\n".join(logs)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        sigma = self.sigma_linear(h)
        if self.instance_label_dimension > 0:
            if self.use_instance_feature_layer:
                instance = self.instance_linear(self.isntance_feature_linear(h))
            else:
                instance = self.instance_linear(h)

        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], dim=-1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        if self.instance_label_dimension > 0:
            outputs = torch.cat([rgb, sigma, instance], -1)
        else:
            outputs = torch.cat([rgb, sigma], -1)

        return outputs

    def load_weights_from_keras(self, weights):
        raise NotImplementedError


def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    input_dirs = viewdirs[:, None].expand(inputs.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = embeddirs_fn(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], dim=-1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_NeRFDecomp(args):
    """
    Instantiate NeRFDecomp Model
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    logger = utils.logging_utils.load_logger("NeRFDecomp Loader")

    input_ch_views = 0
    embeddirs_fn = None
    embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRFDecomp(
        D=args.netdepth, W=args.netwidth, input=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, instance_label_dimension=args.instance_label_dimnension,
        use_instance_feature_layer=args.use_instance_featuer_layer
    ).to(args.device)
    logger.info(model)

    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRFDecomp(

        ).to(args.device)
        logger.info("NeRFDecomp fine model")
        logger.info(model)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        input, viewdirs, network_fn, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk
    )

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load Checkpoint
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join (basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logger.info('Found ckpts: ' + str(ckpts))

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info('Reloading from ' + str(ckpt_path))
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
