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
            self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_instance_label=True,
            instance_label_dimension=0, num_cluster=10, use_basecolor_score_feature_layer=True, use_indirect_feature_layer=True,
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
            use_instance_label: use instance label
            instance_label_dimension: dimension of instance_label
            K: number of base colors
            use_instance_feature_layer: use additional layer following Shuaifeng Zhi et al.
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_instance_label = use_instance_label
        self.instance_label_dimension = instance_label_dimension
        self.num_cluster = num_cluster
        self.use_basecolor_score_feature_layer = use_basecolor_score_feature_layer
        self.use_indirect_feature_layer = use_indirect_feature_layer
        self.use_instance_feature_layer = use_instance_feature_layer

        self.positions_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        self.feature_linear = nn.Linear(W, W)
        self.sigma_linear = nn.Linear(W, 1)

        self.albedo_feature_linear = nn.Linear(W, W // 2)
        self.albedo_linear = nn.Linear(W // 2, 3)

        if self.use_instance_label:
            raise NotImplementedError
        else:
            if self.use_basecolor_score_feature_layer:
                self.basecolor_score_feature_linear = nn.Linear(W, W // 2)
                self.basecolor_score_linear = nn.Linear(W // 2, num_cluster)
            else:
                self.basecolor_score_feature_linear = None
                self.basecolor_score_linear = nn.Linear(W, num_cluster)

        self.direct_linear = nn.Linear(W // 2, 1)
        for k in range(num_cluster):
            if self.use_indirect_feature_layer:
                setattr(self, "indirect_feature_linear{}".format(k), nn.Linear(input_ch_views + W, W // 2))
                setattr(self, "indirect_linear{}".format(k), nn.Linear(W // 2, 1))
            else:
                setattr(self, "indirect_feature_linear{}".format(k), None)
                setattr(self, "indirect_linear{}".format(k), nn.Linear(input_ch_views + W, 1))

    def __str__(self):
        logs = ["[NeRFDecomp"]
        logs += ["\t- depth : {}".format(self.D)]
        logs += ["\t- width : {}".format(self.W)]
        logs += ["\t- input_ch : {}".format(self.input_ch)]
        logs += ["\t- use_instance_label : {}".format(self.use_instance_label)]
        logs += ["\t- instance_label_dimension : {}".format(self.instance_label_dimension)]
        logs += ["\t- use_instance_feature_layer : {}".format(self.use_instance_feature_layer)]
        logs += ["\t- base_color_num : {}".format(self.num_cluster)]
        logs += ["\t- use_indirect_feature_layer : {}".format(self.use_indirect_feature_layer)]
        return "\n".join(logs)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.positions_linears):
            h = self.positions_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        sigma = self.sigma_linear(h)
        albedo_feature = self.albedo_feature_linear(h)
        albedo = self.albedo_linear(albedo_feature)

        if self.use_instance_label:
            if self.instance_label_dimension > 0:
                if self.use_instance_feature_layer:
                    instance = self.instance_linear(self.isntance_feature_linear(h))
                else:
                    instance = self.instance_linear(h)
            raise NotImplementedError

        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], dim=-1)
        
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        # (1)direct illumination
        indirect_illuminations = {}
        for k in range(self.num_cluster):
            if self.use_indirect_feature_layer:
                indirect_illuminations['indirect_illumination{}'.format(k)] = F.relu(getattr(
                    self, 'indirect_feature_linear{}'.format(k)
                )(getattr(
                    self, 'indirect_linear{}'.format(k)
                )(h)))
            else:
                indirect_illuminations['indirect_illumination{}'.format(k)] = F.relu(getattr(
                    self, 'indirect_linear{}'.format(k)
                )(h))

        direct_illumination = F.relu(self.direct_linear(h))


        ret = [sigma, albedo]
        for k in range(self.num_cluster):
            ret.append(indirect_illuminations['indirect_illumination{}'.format(k)])
        ret.append(direct_illumination)
        ret = torch.cat(ret, dim=-1)

        return ret


def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        output = []
        for i in range(0, inputs.shape[0], chunk):
            output_chunk = fn(inputs[i:i + chunk])
            output.append(output_chunk)
        output = torch.cat(output, dim=0)
        return output
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
    skips = [4]
    model = NeRFDecomp(
        D=args.netdepth, W=args.netwidth, input_ch=input_ch, input_ch_views=input_ch_views, skips=skips,
        use_instance_label=args.instance_mask, instance_label_dimension=args.instance_label_dimension,
        num_cluster=args.num_cluster, use_basecolor_score_feature_layer=args.use_basecolor_score_feature_layer,
        use_indirect_feature_layer=args.use_indirect_feature_layer,
        use_instance_feature_layer=args.use_instance_feature_layer
    ).to(args.device)
    logger.info(model)

    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRFDecomp(
            D=args.netdepth, W=args.netwidth, input_ch=input_ch, input_ch_views=input_ch_views, skips=skips,
            use_instance_label=args.instance_mask, instance_label_dimension=args.instance_label_dimension,
            num_cluster=args.num_cluster, use_basecolor_score_feature_layer=args.use_basecolor_score_feature_layer,
            use_indirect_feature_layer=args.use_indirect_feature_layer,
            use_instance_feature_layer=args.use_instance_feature_layer
        ).to(args.device)
        logger.info("NeRFDecomp fine model")
        logger.info(model)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs, viewdirs, network_fn, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk
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

        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'ndc': False,
        'lindisp': args.lindisp
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
