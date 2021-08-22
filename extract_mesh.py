import os, sys
import torch
import numpy as np
import run_nerf
import run_nerf_helpers

import mcubes
import trimesh
import matplotlib.pyplot as plt

#################################################
# marching cube parameters
N = 256
threshold = 0.001
bound = 5.5

#################################################

@ torch.no_grad()
def query(grid_num, bound, chunk, net_query_fn, net_fn):
    t = torch.linspace(-bound, bound, grid_num + 1)
    query_pts = torch.stack(torch.meshgrid(t, t, t), dim=-1).type('torch.cuda.FloatTensor')
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])
    sigma = []
    for i in range(0, flat.shape[0], chunk):
        sigma_chunk = net_query_fn(
            flat[i:i + chunk][..., None, :],
            viewdirs=torch.zeros_like(flat[i: i + chunk]),
            network_fn=net_fn
            )[..., 3]
        sigma.append(sigma_chunk.reshape(-1,))
    sigma = torch.cat(sigma, dim=0)
    sigma = sigma.reshape([*sh[:-1], -1])
    # sigma = torch.maximum(sigma, torch.zeros_like(sigma))
    sigma = sigma.reshape([*sigma.shape[:-1]])
    return sigma
    

def march_cubes(sigma, grid_num, th):
    vertices, triangles = mcubes.marching_cubes(sigma, th)
    mesh = trimesh.Trimesh(vertices / grid_num, triangles)
    return mesh


def main():
    parser = run_nerf.config_parser()
    args = parser.parse_args()
    args.instance_num = 6
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    
    _, render_kwargs_test, _, _, _ = run_nerf.create_nerf(args)

    net_query_fn = render_kwargs_test['network_query_fn']
    net_fn = render_kwargs_test['network_fine']
    sigma = query(N, bound, args.chunk, net_query_fn, net_fn)
    # breakpoint()
    mesh = march_cubes(sigma.cpu().numpy(), grid_num=N, th=threshold)
    mesh.export(os.path.join(basedir, expname, 'mesh_bound={}_th={}.obj'.format(bound, threshold)))


    # test loaded model
    # c2w = np.eye(4)[:3,:4].astype(np.float32)
    """
    c2w = torch.eye(4)[:3,:4].type('torch.cuda.FloatTensor')
    c2w[2, -1] = 11.
    H, W, focal = 800, 800, 875.
    down = 8
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    with torch.no_grad():
        test = run_nerf.render(H//down, W//down, K/down, args.chunk, c2w=c2w, decompose=False, **render_kwargs_test)
    img = np.clip(test[0].cpu().numpy(), 0, 1)
    plt.imshow(img)
    plt.show()
    """



if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()
