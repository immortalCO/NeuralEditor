import contextlib
import sys
import os
import math
from re import M
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import random
from field import ref, spline, sh
import open3d as o3d
import numpy as np

def estimate_norm(pts, param=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
    if param is None:
        cloud.estimate_normals()
    else:
        cloud.estimate_normals(search_param=param)
    
    return torch.from_numpy(np.asarray(cloud.normals)).to(pts.device)

def gaussian_dropout(x, p, training):
    if not training or p == 0:
        return x

    a = (p / (1 - p)) ** 0.5
    return x * (torch.randn_like(x) * a + 1)

def sphere_sym_point(pts, axis):
    dotprod = (pts * axis).sum(dim=-1)
    return 3 * pts - 2 * dotprod.unsqueeze(-1) * axis

def fix_norm_direction(norm, dir):
    # norm: (B, 3)
    # dir: (B, 3)
    # ret: (B, 3)

    norm = norm / denominator(norm.norm(dim=-1, keepdim=True))
    dir = dir / denominator(dir.norm(dim=-1, keepdim=True))
    dotprod = (norm * dir).sum(dim=-1)
    neg = dotprod < 0
    norm[neg] *= -1

    return norm

def norm_axes(z_axis, ref_dir):
    # norm: (B, 3)
    # ret: (B, 3, 3)

    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    ref_dir = ref_dir / ref_dir.norm(dim=-1, keepdim=True)
    dotprod = (z_axis * ref_dir).sum(dim=-1)
    neg = dotprod < 0
    z_axis[neg] *= -1
    dotprod[neg] *= -1
    y_axis = ref_dir - dotprod.unsqueeze(-1) * z_axis

    zero = y_axis.norm(dim=-1) < 1e-4
    if zero.any():
        z_zero = z_axis[zero]
        x, y, z = z_zero.split([1, 1, 1], dim=-1)
        y_can = torch.cat([
            -y-z, x, x,
            y, -x-z, y,
            z, z, -x-y,
        ], dim=-1).view(-1, 3, 3)
        y_cho = z_zero.abs().argmax(dim=-1)
        y_axis[zero] = y_can.gather(dim=-2, index=y_cho[:, None, None].expand(-1, -1, 3)).squeeze(-2)

    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
    x_axis = torch.cross(y_axis, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)

    return torch.stack([x_axis, y_axis, z_axis], dim=-1)
    # the axes are column vectors


def trilinear(pts, const=True):
    x, y, z = pts.split([1, 1, 1], dim=-1)
    tri = [torch.ones_like(x)] if const else []
    tri += [x, y, z, x * y, y * z, x * z, x * y * z]
    tri = torch.cat(tri, dim=-1)

    return tri

def denominator(x, eps=1e-5):
    x[x.abs() < eps] = eps
    return x

def to_density(x):
    return 1 - torch.exp(-x)

def dotprod(x, y):
    return (x * y).sum(dim=-1)

class EmptyEnviron(object):
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

class Recenter(nn.Module):
    def __init__(self, size=[1]):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size, dtype=torch.float).cuda())

    def forward(self, x):
        if self.training:
            mean = x.mean()
            x = x - mean
            with torch.no_grad():
                self.mean.set_(0.9 * self.mean + 0.1 * mean.detach())
        else:
            x = x - self.mean
        return x

class PointNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.dim = dim
        self.bn = nn.BatchNorm1d(dim, affine=affine)
    
    def forward(self, pts):
        # (b, n, dim)
        assert pts.shape[-1] == self.dim
        assert len(pts.shape) == 3

        pts = pts.permute(1, 2, 0)
        # (n, dim, b), so that bn can be applied on n
        pts = self.bn(pts)
        pts = pts.permute(2, 0, 1)
        # restore to (b, n, dim)
        return pts

class Res(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class PosEmbed(nn.Module):
    def __init__(self, rep, mul=2, start=0, include_self=True):
        super().__init__()
        self.rep = rep
        self.mul = mul
        self.start = start
        self.include_self = include_self
        self.dim = 3 * 2 * rep + (3 if include_self else 0)

    def forward(self, x):
        ans = [x] if self.include_self else []
        for i in range(self.start, self.start + self.rep):
            t = math.pi * (self.mul ** i) * x
            ans.append(t.cos())
            ans.append(t.sin())
        return torch.cat(ans, dim=-1)

class PointMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.embed = PosEmbed(2, mul=8, start=1)

        self.pre = nn.Sequential(
            nn.Linear(7 + self.embed.dim + in_dim, 32),
            PointNorm(32),
            nn.ELU(),
            Res(nn.Sequential(
                nn.Linear(32, 32),
                PointNorm(32),
                nn.ELU(),
                nn.Linear(32, 32),
                PointNorm(32)
            )),
            nn.ELU(),
            nn.Linear(32, 64),
            PointNorm(64)
        )
        self.post = nn.Sequential(
            Res(nn.Sequential(
                nn.Linear(64, 64),
                nn.ELU(),
                nn.Linear(64, 64),
            )),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, p, x, conf=None):
        x = torch.cat([trilinear(p, const=False), self.embed(p), x], dim=-1)
        x = self.pre(x)
        
        c = x.softmax(dim=1)
        if conf is not None:
            c = c * conf[:, :, None]
        x = (x * c).sum(dim=1)

        x = self.post(x)
        return x


class Normalize(nn.Module):
    def __init__(self, dim=-1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / denominator(x.norm(dim=self.dim, keepdim=True), self.eps)

class ActivationMov(nn.Module):
    def __init__(self, activate, mov=0):
        super().__init__()
        self.activate = activate
        self.mov = mov
    
    def forward(self, x):
        return self.activate(x + self.mov)

def softhardclamp(x, min=None, max=None):
    xd = x.detach()
    return xd.clamp(min=min, max=max) + (x - xd)

def softclampmax1(x, inplace=False):
    if not inplace:
        x = x.clone()
    mask = x > 1
    x[mask] = 2 - (1 - x[mask]).exp_()
    return x

def softclampmax1_backprop(x, out_grad, inplace=False):
    x_grad = out_grad if inplace else out_grad.clone()
    mask = x > 1
    x_grad[mask] *= (1 - x[mask]).exp_()
    return x_grad

def mat_normalize(x):
    n = x.shape[-1]
    # x = x / x.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-5) * (n ** 0.5)
    norm_inv = x.pow(2).sum(dim=(-1, -2), keepdim=True).clamp(min=1e-8).rsqrt()
    x = x * norm_inv * (n ** 0.5)
    return x


def rand_sim_matrix(n):
    ox = torch.rand(n) * 2 * torch.pi
    oy = torch.rand(n) * 2 * torch.pi
    oz = torch.rand(n) * 2 * torch.pi

    eye = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)
    mx = eye.clone().contiguous()
    mx[:, 1, 1] = mx[:, 2, 2] = torch.cos(ox)
    mx[:, 1, 2] = torch.sin(ox)
    mx[:, 2, 1] = -torch.sin(ox)

    my = eye.clone().contiguous()
    my[:, 0, 0] = my[:, 2, 2] = torch.cos(oy)
    my[:, 0, 2] = -torch.sin(oy)
    my[:, 2, 0] = torch.sin(oy)

    mz = eye.clone().contiguous()
    mz[:, 0, 0] = mz[:, 1, 1] = torch.cos(oz)
    mz[:, 0, 1] = torch.sin(oz)
    mz[:, 1, 0] = -torch.sin(oz)

    flip = eye.clone().contiguous()
    flip[:, 0, 0] = torch.randint(low=0, high=2, size=(n,)).float() * 2 - 1
    flip[:, 1, 1] = torch.randint(low=0, high=2, size=(n,)).float() * 2 - 1
    flip[:, 2, 2] = torch.randint(low=0, high=2, size=(n,)).float() * 2 - 1

    return mx @ my @ mz @ flip

class KDField(nn.Module):
    def __init__(self, pts, embed_dim=32, density_dim=1, field_dim=3, sh_deg=5, spline_samples=8, max_pts_lim=2**21, adjust_height=True, refnerf_mode=True, aggressive_prune=False, resume_immediately=False,
        num_layer=21, num_trees=1, int_height_min=1, int_height_max=4, sample_height_max=5, bnd_pad_fact=1, num_support=96, num_support_nn=32, adaptive_support=True, use_open3d_unit_len=True,
        num_pts=None, pts_embed=None, pts_conf=None, default_field_val=None, pts_clip_grad=False, dropout=0, pts_dropout=0, disable_conf_recenter=True, manual_grad=False, bnd_pad_axis_fact=1,
        warmup_mult=0.5, model_background=False, model_background_R=4, model_background_reg_weight=200, mesh_reg_weight=200, mlp_shapes=[16, 48, 128], reg_samples=1, use_roughness=True, norm_mesh_param=None, norm_mesh_remove_outlier=False, bnd_pad_ignore_bnd=False, grow_img_loss=False,
        render_early_stop=True, train_single_tree=True, assert_check=False):
        # , max_dim=2048, dim_mul_step=3
        super().__init__()

        assert density_dim == 1
        self.max_pts_lim = 0 if max_pts_lim is None else max_pts_lim
        self.set_capacity(num_layer)
        self.trees = None

        

        if norm_mesh_param is None:
            norm_mesh_param =  o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=512)

        self.depth_map_mode = False
        self.model_background = model_background
        self.model_background_R = model_background_R
        self.model_background_reg_weight = model_background_reg_weight
        self.model_background_force_render = False
        self.depth_l = 0
        self.depth_r = 1
        self.regrow_mode = False
        self.integrate_mode = 0
        self.force_norm_pred = False
        self.simple_color_mode = False
        self.naive_plot_mode = False
        self.init_space_transform = True
        self.dropout = dropout
        self.refnerf_mode = refnerf_mode
        self.grow_img_loss = grow_img_loss
        self.aggressive_prune = aggressive_prune
        self.manual_grad = manual_grad
        self.adaptive_support = adaptive_support
        self.adjust_height = adjust_height
        self.use_open3d_unit_len = use_open3d_unit_len
        self.norm_mesh_param = norm_mesh_param
        self.num_support_nn = num_support_nn
        self.norm_mesh_remove_outlier = norm_mesh_remove_outlier
        self.mesh_reg_weight = mesh_reg_weight
        self.warmup_mult = warmup_mult
        self.pts_dropout = pts_dropout
        self.embed_dim = embed_dim
        self.int_height_min = int_height_min
        self.int_height_max = int_height_max
        self.sample_height_max = sample_height_max
        self.bnd_pad_fact = bnd_pad_fact
        self.assert_check = assert_check
        self.num_support = num_support
        self.num_trees = num_trees
        self.deployed_trees = [None] * num_trees
        self.sh_deg = sh_deg
        self.train_single_tree = train_single_tree
        self.num_interpolate_basis = 15
        self.spline_samples = spline_samples
        self.reg_samples = reg_samples
        self.warmup_coef = 1
        self.disable_conf_recenter = disable_conf_recenter
        if default_field_val is None:
            default_field_val = [0] * field_dim
        self.default_field_val = torch.tensor(default_field_val).float().cuda().clamp(min=0.0, max=1.0)

        self.bnd_pad_ignore_bnd = bnd_pad_ignore_bnd

        self.bnd_pad_axis_fact = bnd_pad_axis_fact

        self.original_capacity_config = (num_layer, int_height_min, int_height_max, sample_height_max)

        # self.interpolator = PointMLP(embed_dim, embed_dim * self.num_interpolate_basis)

        self.debug_vars = {}
        self.render_early_stop = render_early_stop

        self.visualize_mode = False
        self.warning_emitted = False

        if num_pts is None:
            if pts is not None:
                num_pts = pts.shape[0]
            else:
                num_pts = self.max_pts
        self.num_pts = num_pts
        
        self.density_dim = density_dim
        self.field_dim = field_dim

        self.render_record_mode = -1
        # -1: disabled
        # 0: record in render_record[0]
        # 1: record in render_record[1]
        # 2: combine [0] and [1] and clear
        self.render_record = [dict(), dict()]
        self.render_record_ratio = 0.5

        if pts_embed is None:
            pts_embed = torch.randn(num_pts, embed_dim)
        
        if pts_embed.shape[1] < embed_dim:
            pts_embed = torch.cat([pts_embed, 
                torch.randn(num_pts, embed_dim - pts_embed.shape[1])
            ], dim=1)
        elif pts_embed.shape[1] > embed_dim:
            pts_embed = pts_embed[:, -embed_dim:]
        assert pts_embed.shape == (num_pts, embed_dim)

        if pts_conf is None:
            pts_conf = torch.full([num_pts], 0.5, dtype=torch.float)
        self.pts_embed_raw = nn.Parameter(pts_embed.clone())
        self.pts_conf_raw = nn.Parameter(pts_conf.clamp(min=1e-6, max=1-1e-6).logit())
        self.pts_embed_norm = nn.BatchNorm1d(embed_dim)
        self.pts_conf_norm = Recenter() if not disable_conf_recenter else nn.Identity()


        if pts_clip_grad:
            self.pts_embed_raw.register_hook(lambda x: x.clamp(min=-1e-2, max=1e-2))
            self.pts_conf_raw.register_hook(lambda x: x.clamp(min=-1e-2, max=1e-2))


        layer1_mlp = lambda n, post=nn.Identity(): nn.Sequential(
            nn.Linear(mlp_shapes[1], n), 
            post
        )

        layer2_mlp = lambda n, post=nn.Identity(): nn.Sequential(
            nn.Linear(mlp_shapes[1], mlp_shapes[0]),
            nn.ELU(),
            nn.Linear(mlp_shapes[0], n),
            post
        )

        
        self.feed_fea = nn.Sequential(
            nn.Linear(embed_dim, mlp_shapes[1]),
            Res(nn.Sequential(
                nn.ELU(),
                nn.Linear(mlp_shapes[1], mlp_shapes[1]),
                nn.ELU(),
                nn.Linear(mlp_shapes[1], mlp_shapes[1]),
            )),
            nn.ELU(),
        )

        self.feed_grad = layer2_mlp(3)
        self.feed_pts_norm = lambda x : -ref.l2_normalize(self.feed_grad(x))
        self.feed_density_noact = layer1_mlp(density_dim)
        self.feed_density_raw = lambda x : F.softplus(self.feed_density_noact(x) + 0.5)

        self.editting = False
        self.enable_space_transform = False
        self.training_st = False
        self.growing = False
        self.fixing_norm_mesh_dir = False
        self.fixed_norm_mesh_dir = False
        self.pts_space_transform = nn.Parameter(torch.zeros(9, device='cuda'))

        if self.refnerf_mode:
            self.feed_diffuse = layer2_mlp(field_dim)
            self.feed_bottleneck = layer1_mlp(mlp_shapes[2], post=nn.ELU())
            self.feed_tint = layer1_mlp(field_dim, post=nn.Sigmoid()) #ActivationMov(nn.Sigmoid(), mov=-0.6931))
            self.feed_roughness = layer1_mlp(1, post=ActivationMov(nn.Softplus(), mov=-1)) if use_roughness else lambda x : None

            self.dir_enc = ref.generate_ide_fn(sh_deg) if use_roughness else ref.generate_dir_enc_fn(sh_deg)
            ref_sh_dim = [0, 4, 10, 20, 38, 72][sh_deg]
            self.feed_specular = nn.Sequential(
                nn.Linear(mlp_shapes[2] + ref_sh_dim + 1, mlp_shapes[1]),
                nn.ELU(),
                layer2_mlp(field_dim),
            )
        else:
            self.feed_field_sh = nn.Sequential( 
                nn.Linear(mlp_shapes[1], mlp_shapes[2]),
                Res(nn.Sequential(
                    nn.ELU(),
                    nn.Linear(mlp_shapes[2], mlp_shapes[2]),
                    nn.ELU(),
                    nn.Linear(mlp_shapes[2], mlp_shapes[2]),
                )),
                nn.ELU(),
                nn.Linear(mlp_shapes[2], field_dim * ((sh_deg + 1) ** 2))
            )

        if self.model_background:
            self.posemb_bg = PosEmbed(16)
            self.feed_bg_bottleneck = nn.Sequential(
                nn.Linear(self.posemb_bg.dim, 128),
                nn.ELU(),
                nn.Linear(128, 256),
                Res(nn.Sequential(
                    nn.ELU(),
                    nn.Linear(256, 256),
                    nn.ELU(),
                    nn.Linear(256, 256),
                )),
                nn.ELU(),
                nn.Linear(256, mlp_shapes[2]),
            )
            self.dir_enc_bg = ref.generate_dir_enc_fn(sh_deg)
            ref_sh_dim_bg = [0, 4, 10, 20, 38, 72][sh_deg]
            self.feed_rgb_bg = nn.Sequential(
                nn.Linear(mlp_shapes[2] + ref_sh_dim_bg, mlp_shapes[1]),
                nn.ELU(),
                layer2_mlp(field_dim)
            )
            self.feed_density_bg = nn.Sequential(
                nn.Linear(mlp_shapes[2], mlp_shapes[1]),
                nn.ELU(),
                layer1_mlp(density_dim, post=nn.Softplus())
            )

        if pts is not None:
            self.init_pts(pts.float().cuda())

    def enable_st(self, reinit=False):
        if reinit or not self.enable_space_transform:
            logging.info("Enable and init space transform")
            self.enable_space_transform = True
            self.pts_space_transform = nn.Parameter(torch.eye(3, device='cuda').reshape(1, 9).expand(self.pts.shape[0], 9).clone())

        if self.naive_plot_mode:
            logging.info("naive plot mode, fixed space transform")
            self.pts_space_transform.requires_grad = False

    def unit_eps(self, fact=1e-3):
        return fact * self.unit_len

    def warmup_step(self, step_per_epoch):
        self.warmup_coef *= self.warmup_mult ** (1 / step_per_epoch)

    def add_reg_loss(self, x, weight=1):
        self.reg_loss = self.reg_loss + x * weight

    def export_cloud(self, file_format=True):
        self.deploy()
        with torch.no_grad():
            pts = self.pts
            embed = self.pts_embed_raw.clone().detach()
            embed = (embed - self.pts_embed_norm.running_mean) / (self.pts_embed_norm.running_var + self.pts_embed_norm.eps).pow(0.5)
            conf = self.pts_conf()

            fea = self.feed_fea(self.pts_embed())
            
            rgb = self.feed_diffuse(fea)
            rgb = (rgb.sigmoid() * (1 + 0.002) - 0.001).clamp(min=0, max=1)

            if not file_format:
                density = to_density(self.feed_density_raw(fea))

        self.undeploy()
        if file_format:
            return {
                'points_with_color': torch.cat([pts, rgb], dim=-1),
                'confidence': conf.reshape(-1),
                'feature': embed
            }
        return pts, rgb, conf, density, embed

    def resume_ckpt(self, ckpt):
        ckpt = torch.load(ckpt)
        s = ckpt['field']
        if 'last_pts_embed' in s:
            del s['last_pts_embed']
        if 'last_pts_conf' in s:
            del s['last_pts_conf']

        # make the shape consistent
        self.pts_embed_raw = nn.Parameter(s['pts_embed_raw'].cuda())
        self.pts_conf_raw = nn.Parameter(s['pts_conf_raw'].cuda())
        if 'pts_space_transform' in s:
            
            self.pts_space_transform = nn.Parameter(s['pts_space_transform'].cuda())
            if len(self.pts_space_transform.shape) != 2:
                logging.info("Empty space transform in ckpt, disable space transform")
                self.enable_space_transform = False
            else:
                logging.info("Resuming space transform")
                self.enable_space_transform = True
        else:
            logging.info("No space transform in ckpt, disable space transform")
            self.enable_space_transform = False
            self.pts_space_transform = nn.Parameter(torch.zeros(9, device='cuda'))
            s['pts_space_transform'] = self.pts_space_transform.data

        self.pts = s['pts'].cuda()
        self.rebuild_trees()

        self.load_state_dict(s)

        self.pts = self.pts.cuda()

        fail_load_tree = False
        for i, tree in enumerate(self.trees):
            if fail_load_tree:
                break
            for j, content in enumerate(tree):
                if fail_load_tree:
                    break
                if isinstance(content, list):
                    for k, item in enumerate(content):
                        # item.set_(s[f"trees_{i}_{j}_{k}"].cuda())
                        if item.shape != s[f"trees_{i}_{j}_{k}"].shape:
                            fail_load_tree = True
                            break
                else:
                    # content.set_(s[f"trees_{i}_{j}"].cuda())
                    if content.shape != s[f"trees_{i}_{j}"].shape:
                        fail_load_tree = True
                        break

        if fail_load_tree:
            logging.warning("Failed to load trees, use rebuilt trees instead")
        else:
            for i, tree in enumerate(self.trees):
                for j, content in enumerate(tree):
                    if isinstance(content, list):
                        for k, item in enumerate(content):
                            item.set_(s[f"trees_{i}_{j}_{k}"].cuda())
                    else:
                        content.set_(s[f"trees_{i}_{j}"].cuda())
        
        if not self.use_open3d_unit_len:
            self.calculate_unit_len()

        if 'epoch' not in ckpt:
            ckpt['epoch'] = -1

        import gc
        gc.collect()
        return ckpt['epoch'] + 1, ckpt


    def assert_no_nan(self, force=False):
        if self.assert_check or force:
            for name, param in self.state_dict().items():
                assert not param.isnan().any(), f"{name} contains nan"

    def pts_embed(self):
        x = self.pts_embed_norm(self.pts_embed_raw)
        self.cur_pts_embed = x
        return x

    def pts_conf(self):
        x = self.pts_conf_raw
        x = self.pts_conf_norm(x)
        x = x.sigmoid()
        x = gaussian_dropout(x, self.pts_dropout, training=self.training)
        self.cur_pts_conf = x
        return x

    def set_capacity(self, num_layer):
        logging.info(f"Set capacity num_layer = {num_layer}")
        self.num_layer = num_layer
        self.max_pts = 2 ** num_layer
        if self.max_pts_lim == 0:
            self.max_pts_lim = self.max_pts

    # def fit_fea_weight(self, node_pts, node_emb, node_conf, interpolator_loss=False):
    #     # if node_pts.shape[1] > self.num_support:
    #     #     ind = torch.randint(low=0, high=node_pts.shape[1], size=[node_pts.shape[0], self.num_support], device='cuda')
    #     #     node_conf = node_conf.gather(1, ind)
    #     #     ind = ind.unsqueeze(-1)
    #     #     node_pts = node_pts.gather(1, ind.expand(-1, -1, node_pts.shape[-1]))
    #     #     node_emb = node_emb.gather(1, ind.expand(-1, -1, node_emb.shape[-1]))
        
    #     # fea_weight = torch.cat([node_pts, node_emb * node_conf.unsqueeze(-1)], dim=-1).reshape(node_pts.shape[0], -1)
    #     fea_weight = (node_pts, node_emb * node_conf.unsqueeze(-1))

    #     return fea_weight

    def query_pts_emb(self, pts, fea_weight, return_intermediate=False, pts_grad=False, cur_pts_embed=None, assume_same=False, from_integrate=False):
        need_norm_dir_reg = from_integrate and (cur_pts_embed is not None) and self.fixing_norm_mesh_dir

        if cur_pts_embed is None:
            cur_pts_embed = self.cur_pts_embed

        if len(pts.shape) <= 2:
            # only (batch, 3)
            pts = pts.unsqueeze(0)
            # (1, batch, 3)
            pts_unsqueezed = True
        else:
            pts_unsqueezed = False

        assert len(fea_weight[0].shape) <= len(pts.shape)

        node_sup = fea_weight
        # (batch, num_support)
        sample, batch, _ = pts.shape
        pts = pts.unsqueeze(-2)

        if not assume_same:
            # only preserve num_support_nn nearest neighbors
            pts_sup = self.pts[node_sup]
            dist_all = torch.cdist(pts, pts_sup.unsqueeze_(0), p=2).squeeze(-2)
            del pts_sup
            # (sample, batch, num_support)
            dist_norm, ind = dist_all.topk(self.num_support_nn, dim=-1, largest=False, sorted=False)
            # (sample, batch, num_support_nn)
            ind_fixed = ind.permute(1, 0, 2).reshape(batch, -1)
            # (batch, sample * num_support_nn, 3)
            node_sup = node_sup.gather(1, ind_fixed)
            # (batch, sample * num_support_nn)
            node_sup = node_sup.reshape(batch, sample, -1).permute(1, 0, 2)
            # (sample, batch, num_support_nn)

            self.cached_query_data = (node_sup, dist_norm)
        else:
            node_sup, dist_norm = self.cached_query_data
        

        dist_fixed = dist_norm + self.unit_eps(fact=1e-3)
        environ = torch.no_grad() if pts_grad else EmptyEnviron()
        with environ:
            node_emb = cur_pts_embed[node_sup] 
            node_conf = self.cur_pts_conf[node_sup]
            node_conf.clamp_(min=1e-5)
            # (sample, batch, num_support_nn, D)

        if return_intermediate:
            node_pts = self.pts[node_sup]
            dist = node_pts.sub_(pts)
            coef_raw = node_conf.log() - dist_fixed.log()
            panelty_raw = self.unit_len_nn / dist_fixed
            panelty = node_conf * softclampmax1(panelty_raw)
            coef_softmax = coef_raw.softmax(dim=-1)
            coef = coef_softmax * panelty
            ans = (node_emb * coef.unsqueeze(-1)).sum(dim=-2)
            return ans, node_emb, node_conf, dist, dist_norm, dist_fixed, coef_raw, coef_softmax, panelty_raw, panelty, coef

        if pts_grad:
            coef_raw = node_conf.log() - dist_fixed.log()
            panelty = node_conf.mul_(softclampmax1(self.unit_len_nn / dist_fixed))
        else:
            panelty = softclampmax1(self.unit_len_nn / dist_fixed).mul_(node_conf)
            coef_raw = node_conf.log() - dist_fixed.log_()
        coef = (coef_raw.softmax(dim=-1) * panelty)
        # (sample, batch, num_support_nn)

        if need_norm_dir_reg:
            sup_mesh = self.pts_norm_mesh[node_sup]
            # (sample, batch, num_support_nn, 3)
            self.norm_dir_reg_data = (coef.clone(), sup_mesh)
        
        ans = node_emb.mul_(coef.unsqueeze_(-1)).sum(dim=-2)

        if pts_unsqueezed:
            ans = ans.squeeze(0)

        return ans


    def integrate(self, line_o, line_d, int_l, int_r, fea_weight, spline_sample_fact=1, cur_pts_embed=None, assume_same=False):
        # line_o, line_d: (B, 3)
        # int_l, int_r: (B)
        # fea_weight: (B, ??) = self.fitfeaweight(node_pts, node_emb)

        # print(f"int func line_o = {line_o.shape} line_d = {line_d.shape} int_l = {int_l.shape} int_r = {int_r.shape}")

        if self.naive_plot_mode or self.integrate_mode == 1:
            pts = line_o + line_d * (int_l + int_r).unsqueeze(-1) / 2
            return self.query_pts_emb(pts, fea_weight, cur_pts_embed=cur_pts_embed, assume_same=assume_same, from_integrate=True)

        samples = int(self.spline_samples * spline_sample_fact)

        if self.integrate_mode == 0:
            x = torch.linspace(0, 1, samples, device='cuda')
        else:
            x = torch.rand(samples, device='cuda').sort()[0]
            x[0] = 0
            x[-1] = 1
        
        qx = torch.tensor([1.], dtype=torch.float, device='cuda')

        pts = line_o.unsqueeze(0) + line_d.unsqueeze(0) * (int_l.unsqueeze(0) + x.unsqueeze(-1) * (int_r - int_l).unsqueeze(0)).unsqueeze(-1)
        # (samples, batch, 3)
        y = self.query_pts_emb(pts, fea_weight, cur_pts_embed=cur_pts_embed, from_integrate=True, assume_same=assume_same)
        # (samples, batch, dim)
        y = y.permute(1, 2, 0)
        # (batch, dim, samples)

        qy = spline.query_int(x, y, qx, assume_uniform_x=True)
        # (batch, dim, 1)
        result = qy.squeeze(-1)

        if self.assert_check:
            assert not result.isnan().any()

        return result

    def arrange(self, pts, rand=False, pca=False, max_pts=None, bnd_pad_fact=None, no_logging=False, no_support=False):
        single_batch = False
        if len(pts.shape) == 2:
            pts = pts.unsqueeze(0)
            single_batch = True

        assert single_batch

        from sklearn.neighbors import NearestNeighbors
        knn_solver = NearestNeighbors(n_neighbors=self.num_support).fit(pts.squeeze(0).cpu().numpy())

        batch = pts.shape[0]
        ind = torch.arange(pts.shape[1], device='cuda')[None, :].expand(batch, -1)

        orig_len = pts.shape[1]
        if max_pts is None:
            max_pts = self.max_pts
        elif max_pts == 0:
            max_pts = 1
            while pts.shape[1] > max_pts:
                max_pts *= 2

        if bnd_pad_fact is None:
            bnd_pad_fact = self.bnd_pad_fact

        assert pts.shape[1] <= max_pts
        if pts.shape[1] < max_pts:
            d = max_pts - pts.shape[1]
            assert d < pts.shape[1], f"tree (max_pts: {max_pts}) too large for {pts.shape[1]} points"

            # extra_ind = torch.randint(low=0, high=pts.shape[1], size=[batch, d], device='cuda')
            extra_ind = torch.stack([
                torch.randperm(pts.shape[1], device='cuda')[:d] for _ in range(batch)
            ], dim=0)
            extra_pts = pts.gather(1, extra_ind.unsqueeze(-1).expand(-1, -1, 3))
            pts = torch.cat([pts, extra_pts], dim=1)
            ind = torch.cat([ind, extra_ind], dim=1)

        assert pts.shape == (batch, max_pts, 3)

        pts = pts.unsqueeze(1)
        ind.unsqueeze_(1)
        
        tree_pts = []
        tree_lay = []
        tree_dir = []
        tree_support = []
        tree_box_d = []
        tree_box_u = []

        box_u = torch.full([pts.size(0), pts.size(1), pts.size(-1)], math.inf, device='cuda')
        box_d = -box_u

        main_axes = torch.eye(3, device='cuda')
        bnd_d, bnd_u = pts.aminmax(dim=2)
        bnd_pad_1 = bnd_pad_fact * (bnd_u - bnd_d) / (orig_len ** 0.5)
        bnd_pad_2 = bnd_pad_fact * torch.tensor([2 * self.unit_len / (3 ** 0.5)] * 3, device='cuda')
        bnd_pad_2 *= self.bnd_pad_axis_fact
        
        if (not self.bnd_pad_ignore_bnd) and (not self.editting):
            bnd_pad = torch.minimum(bnd_pad_1, bnd_pad_2)
        else:
            bnd_pad = bnd_pad_2

        if bnd_pad_fact is self.bnd_pad_fact:
            logging.info(f"set bnd_pad_2 with axis fact {self.bnd_pad_axis_fact}")
            self.bnd_pad = bnd_pad

        

        def fix_box(pts, box_d, box_u, fact=1):
            sub_bnd_d, sub_bnd_u = pts.aminmax(dim=2)

            fbox_d = torch.maximum(box_d, sub_bnd_d - bnd_pad * fact)
            fbox_u = torch.minimum(box_u, sub_bnd_u + bnd_pad * fact)
            
            # for unbounded box, pad by sqrt(2)
            assert self.use_open3d_unit_len
            mask_d = sub_bnd_d - box_d > 64 * self.unit_len
            fbox_d[mask_d] = torch.maximum(box_d[mask_d], (sub_bnd_d - bnd_pad * fact * (2 ** 0.5))[mask_d])
            mask_u = box_u - sub_bnd_u > 64 * self.unit_len
            fbox_u[mask_u] = torch.minimum(box_u[mask_u], (sub_bnd_u + bnd_pad * fact * (2 ** 0.5))[mask_u])

            assert not (fbox_d > fbox_u).any(), f"Error: box_d > box_u: {(fbox_d - fbox_u).max().item()}"

            return fbox_d, fbox_u   

        while pts.shape[2] > 1:
            batch, node, sub, dim = pts.shape

            have_support = (2**self.int_height_min <= pts.shape[2] <= 2**self.sample_height_max) and (not no_support)
            num_support = max(self.num_support, sub * 2) if self.adaptive_support else self.num_support
            if not have_support:
                num_support = 0
           
            tree_pts.append(pts.clone())
            tree_lay.append(ind.clone())

            # fix unbounded boxes
            small_fact = 1 if pts.shape[2] <= 2**self.int_height_max else 2
            fact = min(8, max(1, (small_fact * pts.shape[2] / (2 ** self.int_height_min))**0.5))
            fbox_d, fbox_u = fix_box(pts, box_d, box_u, fact=fact)

            # maintain support
            too_large_cnt = 0
            if have_support:
                # subtree = pts.squeeze(0).mean(dim=-2)
                subtree = (fbox_u + fbox_d).squeeze(0) / 2
                # (node, dim)
                support_dist, support = knn_solver.kneighbors(subtree.cpu().numpy(), n_neighbors=num_support)
                support_dist = torch.tensor(support_dist).amax(dim=-1).cuda()
                support = torch.from_numpy(support).cuda()
                tree_support.append(support)

                # if not no_logging:
                #     pass
                #     logging.debug(f"support_dist max: {support_dist.max().item()}")

                if (2**self.int_height_min <= pts.shape[2] <= 2**self.int_height_max):
                    lim = self.unit_len_box + max(bnd_pad_fact - 1, 0) * self.unit_len * 4
                    too_large = (support_dist > lim).unsqueeze_(0)
                    too_large |= ((fbox_u - fbox_d).norm(dim=-1) > 2 * lim)
                    if too_large.any():
                        too_large_cnt = too_large.sum().item()
                        # swap to disable these boxes
                        tmp_d = fbox_d[too_large]
                        tmp_u = fbox_u[too_large]
                        fbox_d[too_large] = tmp_u
                        fbox_u[too_large] = tmp_d

            else:
                tree_support.append(torch.zeros(node, dtype=torch.long, device='cuda'))
            
            tree_box_d.append(fbox_d)
            tree_box_u.append(fbox_u)

            if not no_logging:
                logging.info(f"arrange level: {len(tree_pts)}, fact: {bnd_pad_fact*fact:.2f} sub: {sub}, #too_large: {too_large_cnt}, num_support: {num_support}")

            


            
            # print(f"build {pts.shape}")
            
            if pca:
                _, _, V = torch.pca_lowrank(pts)
                dir = V[:, :, :, 0]
            elif rand:
                dir = main_axes[torch.randint(low=0, high=3, size=[batch, node], device='cuda')]
            else:
                mode = 0.5 + torch.rand(batch, node, 1, device='cuda').mul_(0.5)
    
                ran = pts.aminmax(dim=-2)
                val0 = ran.max - ran.min
                val1 = pts.std(dim=-2)
                val = val0 * mode + val1 * (1 - mode)

                axis = val.argmax(dim=-1)
                dir = main_axes[axis]
                

            # random flip direction for some nodes
            flip = (-1) ** torch.randint(low=0, high=2, size=[batch, node], device='cuda')[:, :, None]
            dir *= flip
            tree_dir.append(dir)

            val = (pts * dir[:, :, None, :]).sum(dim=-1)
            lval, topk = val.topk(sub // 2, dim=-1, largest=False, sorted=False)
           
            # batch, node, sub//2
            mask = torch.zeros(batch, node, sub, dtype=torch.long, device='cuda')
            mask.scatter_(2, topk, torch.ones_like(topk))
            mask = mask.bool()

            rval = val[~mask].reshape(batch, node, sub // 2)
            bnd = (lval.max(dim=-1).values + rval.min(dim=-1).values) / 2

            if self.assert_check:
                assert (lval.max(dim=-1).values > rval.min(dim=-1).values).sum() == 0

            lch = pts[mask].reshape(batch, node, sub // 2, dim)
            rch = pts[~mask].reshape(batch, node, sub // 2, dim)
            pts = torch.cat([lch, rch], dim=1)

            lind = ind[mask].reshape(batch, node, sub // 2)
            rind = ind[~mask].reshape(batch, node, sub // 2)
            ind = torch.cat([lind, rind], dim=1)

            boxes_d = [box_d.clone(), box_d.clone()]
            boxes_u = [box_u.clone(), box_u.clone()]

            for (k, coef) in [(0, 1), (1, -1)]:
                kdir = dir * coef
                kbnd = bnd.unsqueeze(-1).expand_as(kdir) * coef
                # constraint: pts * kdir <= kbnd
                mask = (kdir > 1e-4)
                boxes_u[k][mask] = torch.minimum(boxes_u[k][mask], kbnd[mask] / kdir[mask])
                mask = (kdir < -1e-4)
                boxes_d[k][mask] = torch.maximum(boxes_d[k][mask], kbnd[mask] / kdir[mask])


            box_d = torch.cat(boxes_d, dim=1)
            box_u = torch.cat(boxes_u, dim=1)
        
        tree_pts.append(pts.clone())
        tree_lay.append(ind.clone())
        # fix unbounded boxes
        fbox_d, fbox_u = fix_box(pts, box_d, box_u)
        tree_box_d.append(fbox_d)
        tree_box_u.append(fbox_u)
        tree_support.append(torch.zeros(max_pts, dtype=torch.long, device='cuda'))
        pts = pts.squeeze(2)
        ind = ind.squeeze(2)
       

        if single_batch:
            pts.squeeze_(0)
            ind.squeeze_(0)

            def apply(a):
                for x in a:
                    x.squeeze_(0)
            apply(tree_pts)
            apply(tree_lay)
            apply(tree_dir)
            apply(tree_support)
            apply(tree_box_d)
            apply(tree_box_u)

        tree = (tree_pts, tree_lay, tree_dir, tree_support, tree_box_d, tree_box_u)
        return tree

    def export_edit_data(self):
        ckpt = dict()
        assert self.fixed_norm_mesh_dir, "norm mesh not fixed"
        if not self.editting:
            ckpt['orig_pts'] = self.pts.cpu()
            ckpt['orig_pts_norm_mesh'] = self.pts_norm_mesh.cpu()
            ckpt['orig_unit_len'] = self.unit_len
            if self.enable_space_transform:
                ckpt['orig_pts_space_transform'] = self.pts_space_transform.cpu()
            else:
                ckpt['orig_pts_space_transform'] = None
        else:
            ckpt['orig_pts'] = self.orig_pts.cpu()
            ckpt['orig_pts_norm_mesh'] = self.orig_pts_norm_mesh.cpu()
            ckpt['orig_unit_len'] = self.orig_unit_len
            if self.enable_space_transform:
                ckpt['orig_pts_space_transform'] = self.orig_pts_space_transform.cpu()
            else:
                ckpt['orig_pts_space_transform'] = None

        ckpt['pts_embed_raw'] = self.pts_embed_raw.detach().cpu()
        ckpt['pts_conf_raw'] = self.pts_conf_raw.detach().cpu()

        for name, module in self.named_modules():
            if name == "":
                continue
            module_dict = dict()
            for k, v in module.state_dict().items():
                module_dict[k] = v.cpu()
            ckpt[name] = module_dict
        
        return ckpt


    def resume_edit_data(self, ckpt, mapping=None, st=None):
        assert self.editting
        self.orig_pts = ckpt['orig_pts'].cuda()
        self.orig_pts_norm_mesh = ckpt['orig_pts_norm_mesh'].cuda()
        self.orig_unit_len = ckpt['orig_unit_len']
        self.orig_pts_space_transform = ckpt['orig_pts_space_transform']
        if self.orig_pts_space_transform is not None:
            self.orig_pts_space_transform = self.orig_pts_space_transform.cuda()

        self.pts_embed_raw = nn.Parameter(ckpt['pts_embed_raw'].cuda())
        self.pts_conf_raw = nn.Parameter(ckpt['pts_conf_raw'].cuda())

        for name, module in self.named_modules():
            if name == "":
                continue
            module_dict = ckpt[name]
            module.load_state_dict(module_dict)
            module.cuda()      

        if mapping is not None:
            self.apply_mapping(mapping, orig_only=True)

        if st is not None:
            self.pts_space_transform = nn.Parameter(st)

    def apply_mapping(self, mapping, orig_only=False, no_build_tree=False):
        # apply mask/ind to the points
        # logging.info(f"mapping applied orig_only = {orig_only}")
        if orig_only:
            assert self.editting
            no_build_tree = True
        with torch.no_grad():
            if not orig_only:
                self.pts = self.pts[mapping]
                self.pts_norm_mesh = self.pts_norm_mesh[mapping]
                if self.enable_space_transform:
                    self.pts_space_transform = self.pts_space_transform[mapping]

            self.pts_embed_raw = nn.Parameter(self.pts_embed_raw[mapping])
            self.pts_conf_raw = nn.Parameter(self.pts_conf_raw[mapping])
            if self.editting:
                self.orig_pts = self.orig_pts[mapping]
                self.orig_pts_norm_mesh = self.orig_pts_norm_mesh[mapping]
                if self.orig_pts_space_transform is not None:
                    self.orig_pts_space_transform = self.orig_pts_space_transform[mapping]

        if not no_build_tree:
            bak_norm_mesh = self.pts_norm_mesh.clone()
            bak_fixed = self.fixed_norm_mesh_dir
            self.init_pts(self.pts)
            self.pts_norm_mesh = bak_norm_mesh
            self.fixed_norm_mesh_dir = bak_fixed


    def init_pts(self, pts, editting=False, axis_fact=None, orig_mapping=None, from_rebuild=False):
        if self.model_background:
            max_R = pts.norm(dim=1).max().item()
            logging.info(f"model_background max_R = {max_R} background R = {self.model_background_R}")
            assert max_R < self.model_background_R, "model_background_R is too small"


        if not from_rebuild:
            if editting:
                logging.info("Scene editting mode")
                
                if not self.editting:
                    assert self.fixed_norm_mesh_dir, "editting mode requires running fix_norm_mesh_dir first"

                    logging.info("Saving original scene")
                    self.orig_pts = self.pts.clone()
                    self.orig_pts_norm_mesh = self.pts_norm_mesh.clone()
                    self.orig_unit_len = self.unit_len
                    if self.enable_space_transform:
                        logging.info("Saving original space transform")
                        self.orig_pts_space_transform = self.pts_space_transform.detach().clone()
                    else:
                        # self.enable_st()
                        self.orig_pts_space_transform = None
                    self.editting = True
                else:
                    logging.info("Skip saving original scene")
                    if self.orig_pts_space_transform is not None:
                        logging.info("restoring original space transform")
                        with torch.no_grad():
                            self.pts_space_transform.set_(self.orig_pts_space_transform)
                    else:
                        self.enable_st(reinit=True)
            else:
                assert not self.editting, "Scene editting mode cannot be disabled"
                self.editting = False
                self.orig_pts = None
                self.orig_pts_norm_mesh = None
                self.orig_pts_space_transform = None

            if orig_mapping is not None:
                assert self.editting
                self.apply_mapping(orig_mapping, orig_only=True)
        else:
            assert editting == self.editting
            assert orig_mapping is None

        logging.info(f"Init pts num_pts = {pts.shape[0]}")

        if axis_fact is not None:
            if not isinstance(axis_fact, torch.Tensor):
                axis_fact = torch.tensor(axis_fact, device='cuda')
            self.bnd_pad_axis_fact = axis_fact
            logging.info(f"Setting bnd_pad_axis_fact = {self.bnd_pad_axis_fact}")
        

        (num_layer, self.int_height_min, self.int_height_max, self.sample_height_max) = self.original_capacity_config
        if self.num_layer != num_layer:
            self.set_capacity(num_layer)

        while pts.shape[0] > self.max_pts:
            self.set_capacity(self.num_layer + 1)
            if self.adjust_height:
                self.int_height_min += 1
                self.int_height_max += 1
                self.sample_height_max += 1
        while pts.shape[0] * 2 <= self.max_pts:
            self.set_capacity(self.num_layer - 1)
            if self.adjust_height:
                self.int_height_min -= 1
                self.int_height_max -= 1
                self.sample_height_max -= 1

        self.int_height_min = max(self.int_height_min, 1)
        self.int_height_max = max(self.int_height_max, self.int_height_min)
        self.sample_height_max = max(self.sample_height_max, self.int_height_max)

        logging.info(f"height adjusted: {self.int_height_min} {self.int_height_max} {self.sample_height_max}")

        if self.trees is not None:
            for i, tree in enumerate(self.trees):
                for j, content in enumerate(tree):
                    if isinstance(content, list):
                        for k, item in enumerate(content):
                            delattr(self, f"trees_{i}_{j}_{k}")
                    else:
                        delattr(self, f"trees_{i}_{j}")

        self.register_buffer("pts", pts)

        if self.use_open3d_unit_len:
            self.calculate_unit_len()

        self.trees = [self.arrange(pts) for _ in range(self.num_trees)]

        for i, tree in enumerate(self.trees):
            for j, content in enumerate(tree):
                if isinstance(content, list):
                    for k, item in enumerate(content):
                        self.register_buffer(f"trees_{i}_{j}_{k}", item)
                else:
                    self.register_buffer(f"trees_{i}_{j}", content)

        self.calculate_norm_mesh()
        if not self.use_open3d_unit_len:
            self.calculate_unit_len()

        if not from_rebuild:
            if self.enable_space_transform:
                self.enable_st(reinit=True)

    def calculate_norm_mesh(self, remove_outlier=None):
        if remove_outlier is None:
            remove_outlier = self.norm_mesh_remove_outlier

        logging.info(f"Calculating norm mesh, remove_outlier = {remove_outlier}")

        self.norm_mesh_complete = not remove_outlier

        if not remove_outlier:
            self.pts_norm_mesh = estimate_norm(self.pts, param=self.norm_mesh_param).float().cuda()
        else:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(self.pts.cpu().numpy())
            cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=64, std_ratio=3.5)
            ind = torch.tensor(np.asarray(ind)).cuda()

            if self.norm_mesh_param is None:
                cloud.estimate_normals()
            else:
                cloud.estimate_normals(search_param=self.norm_mesh_param)

            normals = torch.tensor(np.asarray(cloud.normals)).cuda()
            self.pts_norm_mesh = torch.zeros_like(self.pts).cuda()
            self.pts_norm_mesh[ind] = normals.float()

        self.fixed_norm_mesh_dir = False


    def calculate_unit_len(self):
        if not self.use_open3d_unit_len:
            self.unit_len = []
            for tree in self.trees:
                leaves = tree[0][-1].reshape(2, -1, 3)
                self.unit_len.append((leaves[0] - leaves[1]).norm(dim=-1).mean())
            self.unit_len = sum(self.unit_len) / self.num_trees / 2
        else:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(self.pts.cpu().numpy())
            self.unit_len = np.mean(cloud.compute_nearest_neighbor_distance()).item() / 2

        from sklearn.neighbors import NearestNeighbors
        pts_np = self.pts.squeeze(0).cpu().numpy()
        knn_solver = NearestNeighbors().fit(pts_np)

        dist, _ = knn_solver.kneighbors(pts_np, n_neighbors=self.num_support_nn)
        dist = torch.tensor(dist).amax(dim=-1).cuda()
        self.unit_len_nn = (dist.mean() + dist.std() * 4).item()

        dist, _ = knn_solver.kneighbors(pts_np, n_neighbors=int(self.num_support * 1.05 + 1.5))
        dist = torch.tensor(dist).amax(dim=-1).cuda()
        self.unit_len_box = (dist.mean() + dist.std() * 5).item()
        logging.info("unit_len: %f, unit_len_nn: %f, unit_len_box: %f", self.unit_len, self.unit_len_nn, self.unit_len_box)
            

    def rebuild_trees(self, num_trees=None):
        if num_trees is not None:
            self.num_trees = num_trees
        self.init_pts(self.pts, editting=self.editting, from_rebuild=True)

    def intersect_box(self, line_o, line_d, box_d, box_u, fetch_range=False, positive_only=True):

        bad = (box_d > box_u).any(dim=-1)
        # swap
        tmp_d = box_d[bad]
        tmp_u = box_u[bad]
        box_d[bad] = tmp_u
        box_u[bad] = tmp_d

        inv_d = 1 / line_d
        A = (box_d - line_o) * inv_d
        B = (box_u - line_o) * inv_d

        def fmx(x):
            x = x.clone()
            x[x.isnan()] = -math.inf
            return x
        def fmn(x):
            x = x.clone()
            x[x.isnan()] = math.inf
            return x

        def pwmin(A, B):
            x = torch.minimum(fmn(A), fmn(B))
            x[A.isnan() & B.isnan()] = math.nan
            return x
        def pwmax(A, B):
            x = torch.maximum(fmx(A), fmx(B))
            x[A.isnan() & B.isnan()] = math.nan
            return x


        pmin = pwmin(A, B)
        pmax = pwmax(A, B)

        vmin = fmx(pmin).max(dim=-1).values
        vmax = fmn(pmax).min(dim=-1).values

        if positive_only:
            vmin = vmin.clamp(min=0)

        intersect = vmin + self.unit_eps(1/16) < vmax

        if fetch_range:
            return intersect, bad, vmin, vmax

        return intersect
        
    def render(self, tree, line_o, line_d, pts_embed, pts_conf, sample_per_box=0, deployed_tree=None):
        import copy
        self.assert_no_nan()

        if self.render_record_mode != -1:
            assert not self.training, "render_record must be disabled in training mode"
            assert sample_per_box == 0, "render_record must be disabled in sampling mode"

        num_line, _ = line_o.shape
        que_line = torch.arange(num_line, device='cuda')
        que_node = torch.zeros(num_line, dtype=torch.long, device='cuda')
        pieces = []

        (tree_pts, tree_lay, _, tree_support, tree_box_d, tree_box_u) = tree

        # pts_embed: (N, emb_dim)

        # print(f"forward")

        intersect_box_res = None

        for i in range(self.num_layer - self.int_height_min + 1):
            height = self.num_layer - i

            lay_pts = tree_pts[i]
            lay = tree_lay[i]
            box_d = tree_box_d[i]
            box_u = tree_box_u[i]
            support = tree_support[i]
            # lay: (nodes, sub)
            # pts: (nodes, sub, 3)
            # box_d, box_u: (nodes, 3)

            intersect_box_res = self.intersect_box(
                line_o[que_line], line_d[que_line], box_d[que_node], box_u[que_node], fetch_range=True)

            mask, bad, que_int_l, que_int_r = intersect_box_res
            intersect_box_res = None
            que_good_box = ~bad[mask]
            que_line = que_line[mask]
            que_node = que_node[mask]
            que_int_l = que_int_l[mask]
            que_int_r = que_int_r[mask]

            # print(f"layer #{i} #que = {que_line.shape[0]} #lay = {lay.shape}")
            if que_line.shape[0] == 0:
                break

            def query_on_line(que_line, que_node, que_int_l, que_int_r, sample_only=False, spline_sample_fact=1):
                if self.visualize_mode:
                    print(f"query_on_line #{i} que_node = {que_node.shape} que_int_l = {que_int_l.shape} que_int_r = {que_int_r.shape}")
                    self.debug_vars['int_boxes'].append((box_d[que_node], box_u[que_node], que_int_l.clone(), que_int_r.clone()))

                
                que_line_o = line_o[que_line]
                que_line_d = line_d[que_line]

                sup_que_node = support[que_node] #[:, :self.num_support]
                # que_sup_embed = pts_embed[sup_que_node]
                # fea_weight = self.fit_fea_weight(self.pts[sup_que_node], que_sup_embed, pts_conf[sup_que_node])
                fea_weight = sup_que_node

                # if self.visualize_mode:
                #     print(f"fea_weight = {fea_weight.shape}")

                if not sample_only:
                    que_fea = self.integrate(
                                que_line_o, que_line_d, que_int_l, que_int_r, fea_weight,
                                spline_sample_fact=spline_sample_fact)
                    que_len = (que_int_r - que_int_l).unsqueeze(-1) / self.unit_len
                    que_fea = que_fea / que_len

                    que_fed = self.feed_fea(que_fea)
                    
                    
                    que_density_raw = self.feed_density_raw(que_fed) * que_len
                    if self.training and not self.training_st:
                        self.add_reg_loss(torch.log(1 + 2 * que_density_raw.pow(2)).sum())
                    # que_density = to_density(que_density_raw)
                    que_density = que_density_raw

                    # if que_density.min() < 1e-4:
                    #     mask = (que_density >= 1e-4).reshape(-1)
                    #     que_line = que_line[mask]
                    #     que_node = que_node[mask]
                    #     que_int_l = que_int_l[mask]
                    #     que_int_r = que_int_r[mask]

                    #     que_line_o = que_line_o[mask]
                    #     que_line_d = que_line_d[mask]
                    #     fea_weight = sup_que_node = sup_que_node[mask]
                        
                    #     que_len = que_len[mask]   
                    #     que_fed = que_fed[mask]     
                    #     que_density = que_density[mask]     

                    que_norm_mesh = ref.l2_normalize(self.integrate(que_line_o, que_line_d, que_int_l, que_int_r, fea_weight,
                                                    spline_sample_fact=spline_sample_fact, cur_pts_embed=self.pts_norm_mesh, assume_same=True))

                    if self.enable_space_transform:
                        st = self.integrate(que_line_o, que_line_d, que_int_l, que_int_r, fea_weight,
                                            spline_sample_fact=spline_sample_fact, cur_pts_embed=self.pts_space_transform, assume_same=True)
                        st = mat_normalize(st.reshape(*st.shape[:-1], 3, 3))

                    if (self.training and not self.training_st) or self.force_norm_pred:
                        que_norm_pred = self.feed_pts_norm(que_fed)
                        if self.enable_space_transform:
                            que_norm_pred = ref.l2_normalize(que_norm_pred.unsqueeze(-2).matmul(st.transpose(-1, -2)).squeeze(-2))

                        if self.force_norm_pred:
                            que_norm = que_norm_pred
                        else:
                            que_norm = ref.l2_normalize(que_norm_pred * self.warmup_coef + que_norm_mesh * (1 - self.warmup_coef))
                    else:
                        que_norm = que_norm_mesh

                    que_fed = gaussian_dropout(que_fed, self.dropout, training=self.training)

                    if not self.training_st:
                        que_dir_ref = ref.reflect(-que_line_d, que_norm)

                        if self.enable_space_transform:
                            # Use st to transform que_dir_ref back to original scene's system
                            que_dir_ref = ref.l2_normalize(que_dir_ref.unsqueeze(-2).matmul(st).squeeze(-2))
                        
                        if self.depth_map_mode:
                            b = que_int_l.unsqueeze(0)
                            k = (que_int_r - que_int_l).unsqueeze(0)
                            num_samples = int(self.spline_samples * spline_sample_fact + 0.5)
                            sample_t = b + k * torch.linspace(0, 1, num_samples, device='cuda').unsqueeze(-1)
                            # (num_samples, batch)

                            sample_pts = que_line_o[None, :, :] + que_line_d[None, :, :] * sample_t[:, :, None]
                            # (num_samples, batch, 3)
                            sample_emb = self.query_pts_emb(sample_pts, fea_weight)
                            sample_emb_fed = self.feed_fea(sample_emb)
                            sample_den = to_density(self.feed_density_raw(sample_emb_fed))

                            sample_den, ind = sample_den.squeeze(-1).max(dim=0)
                            depth_t = sample_t.gather(0, ind[None, :].expand(1, sample_t.shape[1])).squeeze(0)

                            # print("debug depth:, ", depth_t.aminmax(), flush=True)

                            depth_t = (self.depth_r - depth_t) / (self.depth_r - self.depth_l)
                            que_field = depth_t.unsqueeze(-1).repeat(1, self.field_dim)
                            # (batch, field_dim)

                        else:
                            if self.refnerf_mode:
                                if self.render_record_mode != 2:
                                    que_diffuse = self.feed_diffuse(que_fed)
                                    que_tint = self.feed_tint(que_fed)
                                    que_bottleneck = self.feed_bottleneck(que_fed)
                                    que_roughness = self.feed_roughness(que_fed)
                                    que_bottleneck = gaussian_dropout(que_bottleneck, self.dropout, training=self.training)

                                    if self.simple_color_mode:
                                        # que_tint = torch.full_like(que_tint, 0.5)
                                        que_norm = que_norm.detach()
                                        que_tint = que_tint.detach()
                                        que_roughness = torch.zeros_like(que_roughness)

                                    que_enc = self.dir_enc(que_dir_ref, que_roughness)
                                    que_dotprod = (que_line_d * que_norm).sum(-1, keepdim=True)
                                    que_specular = self.feed_specular(torch.cat([que_bottleneck, que_enc, que_dotprod], dim=-1))

                                    if self.render_record_mode != -1:
                                        ind = self.render_record_mode
                                        self.render_record[ind][height] = (que_density, que_diffuse, que_specular, que_tint)
                                else:
                                    que_density_0, que_diffuse_0, que_specular_0, que_tint_0 = self.render_record[0][height]
                                    que_density_1, que_diffuse_1, que_specular_1, que_tint_1 = self.render_record[1][height]

                                    r = self.render_record_ratio
                                    que_density = que_density_0 * (1 - r) + que_density_1 * r
                                    que_diffuse = que_diffuse_0 * (1 - r) + que_diffuse_1 * r
                                    que_specular = que_specular_0 * (1 - r) + que_specular_1 * r
                                    que_tint = que_tint_0 * (1 - r) + que_tint_1 * r

                                    del self.render_record[0][height]
                                    del self.render_record[1][height]

                                que_field_lin = que_diffuse * 0.5 + que_specular * que_tint
                            else:
                                assert False, "unsupported"
                                que_field_sh = self.feed_field_sh(que_fed).reshape(-1, self.field_dim, (self.sh_deg + 1) ** 2)
                                que_field_lin = sh.interpolate(self.sh_deg, que_field_sh, que_line_d) #que_dir_ref)

                            que_field = que_field_lin.sigmoid() * (1 + 0.002) - 0.001   

                            if not self.training:
                                que_field = que_field.clamp(min=1e-8, max=1-1e-8)      

                    else:
                        que_field = torch.zeros(que_line.shape[0], self.field_dim, device='cuda')

                    if self.assert_check:
                        assert not que_fed.isnan().any()
                        assert not que_int_l.isnan().any()
                        assert not que_int_r.isnan().any()
                        assert not que_density.isnan().any()
                        if not self.training_st:
                            if self.refnerf_mode:
                                assert not que_diffuse.isnan().any()
                                assert not que_tint.isnan().any()
                                assert not que_bottleneck.isnan().any()
                                assert not que_roughness.isnan().any()
                                assert not que_enc.isnan().any()
                                assert not que_dotprod.isnan().any()
                                assert not que_specular.isnan().any()
                            else:
                                assert False, "unsupported"
                                assert not que_field_sh.isnan().any()
                            assert not que_field_lin.isnan().any()
                            assert not que_field.isnan().any()

                    # make density grad and norm close
                    if self.training:

                        if not self.fixing_norm_mesh_dir:
                            if not self.training_st:
                                norm_dir_reg = (que_norm_pred * que_line_d).sum(dim=-1).clamp(min=0).pow(2)
                                norm_mesh_reg = (1 - (que_norm_mesh * que_norm_pred).sum(dim=-1))
                            else:
                                que_norm_st = ref.l2_normalize(self.integrate(que_line_o, que_line_d, que_int_l, que_int_r, fea_weight,
                                                    spline_sample_fact=spline_sample_fact, cur_pts_embed=self.orig_pts_norm_mesh, assume_same=True))
                                que_norm_st = ref.l2_normalize(que_norm_st.unsqueeze(-2).matmul(st.transpose(-1, -2)).squeeze(-2))
                                norm_dir_reg = (que_norm_st * que_line_d).sum(dim=-1).clamp(min=0).pow(2)
                                norm_mesh_reg = (1 - (que_norm_mesh * que_norm_st).sum(dim=-1))
                        else:
                            assert self.norm_dir_reg_data is not None
                            coef, que_mesh_norm = self.norm_dir_reg_data
                            # coef : (sample, batch, num_support_nn)
                            # que_mesh_norm : (sample, batch, num_support_nn, 3)

                            que_mesh_loss = (que_mesh_norm * que_line_d.unsqueeze(-2)).sum(dim=-1) #.clamp(min=0).pow(2)
                            # (sample, batch, num_support_nn)
                            norm_dir_reg = (que_mesh_loss * coef).sum(dim=-1).mean(dim=0)
                            # (batch,)

                            norm_mesh_reg = torch.zeros((que_norm.shape[0]), device=que_norm.device)

                            self.norm_dir_reg_data = None
                        
                        if not self.training_st:
                            sample_t = que_int_l + (que_int_r - que_int_l) * torch.rand(self.reg_samples, que_line.shape[0], device='cuda')
                            sample_pts = que_line_o[None, :, :] + que_line_d[None, :, :] * sample_t[:, :, None]
                            sample_pts_fed = self.feed_fea(self.query_pts_emb(sample_pts, fea_weight))

                            grad_pred = ref.l2_normalize(self.feed_grad(sample_pts_fed))
                            if self.enable_space_transform:
                                st = self.query_pts_emb(sample_pts, fea_weight, cur_pts_embed=self.pts_space_transform)
                                st = mat_normalize(st.reshape(*st.shape[:-1], 3, 3))
                                grad_pred = ref.l2_normalize(grad_pred.unsqueeze(-2).matmul(st.transpose(-1, -2)).squeeze(-2))
                            
                            if self.assert_check:
                                assert not grad_pred.isnan().any()

                            # manual grad
                            
                            if self.manual_grad:
                                # assert False, "Not updated"

                                embed, node_emb, node_conf, dist, dist_norm, dist_fixed, coef_raw, coef_softmax, panelty_raw, panelty, coef = self.query_pts_emb(
                                    sample_pts, fea_weight, return_intermediate=True)
                                embed = embed.detach()

                                mlp1 = self.feed_fea[0]
                                mlp2 = self.feed_fea[1].module[1]
                                mlp3 = self.feed_fea[1].module[3]
                                mlp4 = self.feed_density_noact[0]
                                with torch.no_grad():
                                    mlp1_out = mlp1(embed)
                                    res_elu1_out = F.elu(mlp1_out)
                                    res_mlp2_out = mlp2(res_elu1_out)
                                    res_elu2_out = F.elu(res_mlp2_out)
                                    res_mlp3_out = mlp3(res_elu2_out)
                                    elu3_out = F.elu(mlp1_out + res_mlp3_out)
                                    density = mlp4(elu3_out)

                                    
                                def mlp_backprop(mlp, out_grad):
                                    return out_grad.matmul(mlp.weight)
                                
                                def elu_backprop(elu_out, out_grad, inplace=False):
                                    if inplace:
                                        return out_grad.mul_(elu_out.clamp_(max=0).add_(1))
                                    return out_grad * (elu_out.clamp_(max=0).add_(1))

                                def batched_diag(S):
                                    return S.unsqueeze(-1) * torch.eye(S.shape[-1], device=S.device)

                                def batched_outer(S):
                                    return S.unsqueeze(-1) * S.unsqueeze(-2)
                                
                                density_grad = torch.ones_like(density)
                                elu3_out_grad = mlp_backprop(mlp4, density_grad)
                                res_mlp3_out_grad = elu_backprop(elu3_out, elu3_out_grad)
                                res_elu2_out_grad = mlp_backprop(mlp3, res_mlp3_out_grad)
                                res_mlp2_out_grad = elu_backprop(res_elu2_out, res_elu2_out_grad, inplace=True)
                                res_elu1_out_grad = mlp_backprop(mlp2, res_mlp2_out_grad)
                                mlp1_out_grad = elu_backprop(res_elu1_out, res_elu1_out_grad, inplace=True).add_(res_mlp3_out_grad)
                                embed_grad = mlp_backprop(mlp1, mlp1_out_grad)

                                # embed = (node_emb * coef.unsqueeze(-1)).sum(dim=-2)
                                coef_grad = embed_grad.unsqueeze_(-2).mul(node_emb).sum(dim=-1)
                                # coef = coef_softmax * panelty
                                coef_softmax_grad = panelty * coef_grad
                                panelty_grad = coef_softmax * coef_grad
                                # coef_softmax = coef_raw.softmax(dim=-1)
                                jacob_coef = batched_diag(coef).sub_(batched_outer(coef))
                                coef_raw_grad = coef_softmax_grad.unsqueeze_(-2).matmul(jacob_coef).squeeze_(-2).mul_(panelty)
                                # panelty = node_conf * softclampmax1(panelty_raw)
                                panelty_raw_grad = softclampmax1_backprop(panelty_raw, panelty_grad.mul_(node_conf), inplace=True)
                                # panelty_raw = self.unit_len_nn / dist_fixed
                                dist_fixed_grad_1 = panelty_raw_grad.div_(dist_fixed.pow(2)).mul_(-self.unit_len_nn)
                                # coef_raw = node_conf.log() - dist_fixed.log()
                                dist_fixed_grad_2 = coef_raw_grad.neg_().div_(dist_fixed)
                                dist_fixed_grad = dist_fixed_grad_1.add_(dist_fixed_grad_2)
                                # dist_fixed = dist_norm + self.unit_eps(fact=1e-3)
                                dist_norm_grad = dist_fixed_grad
                                # dist_norm = dist.norm(dim=-1)
                                dist_grad = dist.mul_(dist_norm_grad.div_(denominator(dist_norm, eps=self.unit_eps())).unsqueeze_(-1))
                                # dist = node_pts.sub_(pts)
                                pts_grad = dist_grad.neg_()

                                grad_real = ref.l2_normalize(pts_grad.sum(dim=-2))

                                if self.assert_check:
                                    assert not density_grad.isnan().any()
                                    assert not elu3_out_grad.isnan().any()
                                    assert not res_mlp3_out_grad.isnan().any()
                                    assert not res_elu2_out_grad.isnan().any()
                                    assert not res_mlp2_out_grad.isnan().any()
                                    assert not res_elu1_out_grad.isnan().any()
                                    assert not mlp1_out_grad.isnan().any()
                                    assert not embed_grad.isnan().any()

                                    assert not coef_grad.isnan().any()
                                    assert not coef_raw_grad.isnan().any()
                                    assert not dist_fixed_grad.isnan().any()
                                    assert not dist_norm_grad.isnan().any()
                                    assert not dist_grad.isnan().any()
                                    assert not pts_grad.isnan().any()

                            else:
                                def detached(model):
                                    model = copy.deepcopy(model)
                                    for param in model.parameters():
                                        param.requires_grad_(False)
                                    return model
                                
                                dtch_feed_fea = detached(self.feed_fea)
                                dtch_feed_density = detached(self.feed_density_noact)

                                sample_pts.requires_grad_(True)
                                density = dtch_feed_density(dtch_feed_fea(self.query_pts_emb(sample_pts, fea_weight, pts_grad=True)))
                                density.sum().backward()
                                grad_real = ref.l2_normalize(sample_pts.grad)
                                sample_pts.requires_grad_(False)


                            if self.assert_check:
                                assert not grad_real.isnan().any()

                            # # grad_pred/grad_real: (sample, batch, 3)             
                            # fix = lambda norm: norm.permute(1, 0, 2).reshape(-1, 3 * self.reg_samples)
                            # # after fix: (batch, 3 * sample)
                            # que_norm_grad = torch.cat([que_norm, que_line_d, fix(grad_pred), fix(grad_real)], dim=-1)

                            
                            grad_pred_reg = (1 - (grad_pred * grad_real).sum(dim=-1)).mean(dim=0)
                        else:
                            grad_pred_reg = torch.zeros((que_norm.shape[0]), device=que_norm.device)

                        if self.assert_check:
                            assert not norm_dir_reg.isnan().any()
                            assert not grad_pred_reg.isnan().any()
                            assert not norm_mesh_reg.isnan().any()

                        que_norm = torch.stack([norm_dir_reg, grad_pred_reg, norm_mesh_reg], dim=-1)

                    else:
                        que_norm = torch.ones((que_norm.shape[0], 1), device=que_norm.device)

                    pieces.append([que_line, que_int_l, que_int_r, que_density, que_field, que_norm])

                if sample_per_box > 0:
                    training = self.training
                    if training:
                        self.eval()
                    with torch.no_grad():
                        b = que_int_l.unsqueeze(0)
                        k = (que_int_r - que_int_l).unsqueeze(0)
                        sample_t1 = b + k * torch.linspace(0, 1, sample_per_box, device='cuda').unsqueeze(-1)
                        sample_t2 = b + k * torch.rand(sample_per_box // 2, b.shape[1], device='cuda')
                        sample_t = torch.cat([sample_t1, sample_t2], dim=0)

                        sample_pts = que_line_o[None, :, :] + que_line_d[None, :, :] * sample_t[:, :, None]
                        sample_emb = self.query_pts_emb(sample_pts, fea_weight)
                        sample_emb_fed = self.feed_fea(sample_emb)
                        sample_den = to_density(self.feed_density_raw(sample_emb_fed))

                        sample_den, ind = sample_den.squeeze(-1).max(dim=0)
                        sample_pts = sample_pts.gather(0, ind[None, :, None].expand(1, sample_pts.shape[1], sample_pts.shape[2])).squeeze(0)
                        sample_emb = sample_emb.gather(0, ind[None, :, None].expand(1, sample_emb.shape[1], sample_emb.shape[2])).squeeze(0)

                        sample_dis1 = ((sample_pts[:, None, :] - self.pts[sup_que_node]).norm(dim=2) 
                                        / pts_conf[sup_que_node].clamp(min=1e-5)).min(dim=1).values / self.unit_len
                        sample_dis2 = ((sample_pts[:, None, :] - lay_pts[que_node]).norm(dim=2) 
                                        / pts_conf[lay[que_node]].clamp(min=1e-5)).min(dim=1).values / self.unit_len
                        sample_dis = torch.minimum(sample_dis1, sample_dis2)

                        if self.enable_space_transform:
                            sample_st = mat_normalize(self.query_pts_emb(sample_pts, fea_weight, cur_pts_embed=self.pts_space_transform))
                            sample_emb = torch.cat([sample_emb, sample_st], dim=-1)

                        # undo batchnorm
                        # pen = self.pts_embed_norm
                        # sample_emb = (sample_emb - pen.bias) / pen.weight
                        # sample_emb = sample_emb * (pen.running_var + pen.eps).pow(0.5) + pen.running_mean

                        self.sample_pts.append(torch.cat([que_line.unsqueeze(-1).float(), sample_pts, sample_emb, sample_den.unsqueeze(-1), sample_dis.unsqueeze(-1)], dim=-1))
                    if training:
                        self.train()
                    assert training == self.training

            if i == self.num_layer - self.int_height_min:
                if que_good_box.any():
                    query_on_line(que_line[que_good_box], que_node[que_good_box], que_int_l[que_good_box], que_int_r[que_good_box])
                break

            # Step -1: Recursively query children nodes
            last_que_line, last_que_node = que_line, que_node
            que_line = torch.cat([last_que_line, last_que_line], dim=-1)
            que_node = torch.cat([last_que_node, last_que_node + lay.shape[0]], dim=-1)

            if i >= self.num_layer - (self.sample_height_max if sample_per_box > 0 else self.int_height_max):
                intersect_box_res = self.intersect_box(
                    line_o[que_line], line_d[que_line], tree_box_d[i + 1][que_node], tree_box_u[i + 1][que_node], fetch_range=True)

                sample_only = not (i >= self.num_layer - self.int_height_max)

                removed = ~intersect_box_res[0].reshape(2, -1)
                removed = (removed[0] & removed[1]) & que_good_box
                # removed: will be removed at next layer, info loss
                if removed.any():
                    # print(f"#removed = {removed.sum()}")
                    spline_sample_fact = 2**(0.5 * min(3, self.num_layer - self.int_height_min - i))
                    assert spline_sample_fact > 1
                    query_on_line(
                        last_que_line[removed], last_que_node[removed], que_int_l[removed], que_int_r[removed], 
                        sample_only=sample_only, spline_sample_fact=spline_sample_fact)

                if sample_per_box > 0 and not sample_only:
                    preserve = ~removed & que_good_box
                    if preserve.any():
                        query_on_line(
                            last_que_line[preserve], last_que_node[preserve], que_int_l[preserve], que_int_r[preserve], 
                            sample_only=True)

        
        if len(pieces) == 0:
            ans = torch.zeros([num_line, self.field_dim], dtype=torch.float, device='cuda') 
            den = torch.zeros([num_line, 1], dtype=torch.float, device='cuda')
            return torch.cat([ans, den], dim=-1)

        # Last Step: in batches, calculate each ray's answer
        que_line = []
        que_int_l = []
        # que_int_r = []
        # que_density = []
        # que_field = []
        # que_norm = []
        que_stuff = []
        for line, int_l, int_r, density, field, norm in pieces:
            que_line.append(line)
            que_int_l.append(int_l)
            # que_int_r.append(int_r)
            # que_density.append(density)
            # que_field.append(field)
            # que_norm.append(norm)
            que_stuff.append(torch.cat([density, field, norm], dim=-1))
        que_line = torch.cat(que_line, dim=0)
        que_int_l = torch.cat(que_int_l, dim=0)
        # # que_int_r = torch.cat(que_int_r, dim=0)
        # que_density = torch.cat(que_density, dim=0)
        # que_field = torch.cat(que_field, dim=0)
        # que_norm = torch.cat(que_norm, dim=0)
        que_stuff = torch.cat(que_stuff, dim=0)

        count = torch.zeros(num_line, dtype=torch.long, device='cuda')
        count.scatter_add_(0, que_line, torch.ones_like(que_line))
        max_count = count.max().item()

        zero_count = (count == 0).sum().item()
        if zero_count > 0:
            if not self.warning_emitted:
                logging.debug(f"{zero_count} rays are not intersected with any box")
                self.warning_emitted = True

        nonzero = count > 0
        nonzero_ind = torch.arange(num_line, device='cuda')[nonzero]
        num_nonzero_line = nonzero_ind.shape[0]
        nonzero_map = torch.full([num_line], -1, dtype=torch.long, device='cuda')
        nonzero_map[nonzero_ind] = torch.arange(num_nonzero_line, device='cuda')
        que_nonzero = nonzero[que_line]

        count = count[nonzero]
        line_o = line_o[nonzero]
        line_d = line_d[nonzero]
        que_line = nonzero_map[que_line[que_nonzero]]
        que_int_l = que_int_l[que_nonzero]
        # # que_int_r = que_int_r[que_nonzero]
        # que_density = que_density[que_nonzero]
        # que_field = que_field[que_nonzero]
        # que_norm = que_norm[que_nonzero]
        que_stuff = que_stuff[que_nonzero]

        if self.assert_check:
            assert que_line.min() >= 0
            assert que_line.max() < num_nonzero_line
        
        _, ind1 = que_int_l.sort(descending=False)
        _, ind2 = que_line[ind1].sort(descending=False, stable=True)
        ind = ind1[ind2]
        que_line = que_line[ind]
        # que_density = que_density[ind]
        # que_field = que_field[ind]
        # que_norm = que_norm[ind]
        que_stuff = que_stuff[ind]

        count_cumsum = count.cumsum(dim=0)
        first_pos = torch.cat([torch.tensor([0], device='cuda'), count_cumsum[:-1]], dim=0)
        last_pos = count_cumsum - 1

        # que_ans = torch.zeros([num_nonzero_line, self.field_dim], dtype=torch.float, device='cuda')
        # que_cumprod = torch.ones([num_nonzero_line, self.density_dim], dtype=torch.float, device='cuda')

        norm_dim = que_stuff.shape[-1] - self.density_dim - self.field_dim
        que_density_raw, que_field, que_norm = que_stuff.split([self.density_dim, self.field_dim, norm_dim], dim=-1)

        
        log_rest = -que_density_raw
        rest = log_rest.exp()
        que_density = 1 - rest

        # Do in double
        log_cumprod = torch.cat([torch.tensor([[0.]]).double().cuda(), log_rest[:-1].double().cumsum_(dim=0)], dim=0)
        log_cumprod = log_cumprod - log_cumprod[first_pos[que_line]]
        cumprod = log_cumprod.exp_().float()
        # Back to float
        que_cumprod = cumprod[last_pos] * rest[last_pos]
        
        que_coef = cumprod * que_density
        que_ans = torch.zeros([num_nonzero_line, self.field_dim], dtype=torch.float, device='cuda')
        que_ans.scatter_add_(0, que_line[:, None].repeat(1, self.field_dim), que_coef * que_field)

        if self.assert_check:
            assert 0 <= rest.min()
            assert rest.max() <= 1
            assert 0 <= que_density.min()
            assert que_density.max() <= 1
            assert 0 <= que_coef.min()
            assert que_coef.max() <= 1

        if self.training:
            norm_dir_reg, grad_pred_reg, norm_mesh_reg = (que_norm * que_coef).sum(dim=0)
            self.add_reg_loss(norm_dir_reg, weight=10000)
            self.add_reg_loss(grad_pred_reg, weight=30)
            self.add_reg_loss(norm_mesh_reg, weight=min(self.mesh_reg_weight, max(1, self.mesh_reg_weight * self.warmup_coef / self.warmup_mult)))


        ans = torch.zeros([num_line, self.field_dim], dtype=torch.float, device='cuda') 
        den = torch.zeros([num_line, self.density_dim], dtype=torch.float, device='cuda')
        # if not self.training:
        #     ans -= 1000000000
        ans[nonzero_ind] = que_ans / (1 - que_cumprod).clamp(min=1e-6)
        den[nonzero_ind] = 1 - que_cumprod

        line_intersect = torch.zeros(num_line, dtype=torch.bool, device='cuda')
        line_intersect[nonzero_ind] = True
        self.line_intersect &= line_intersect

        if self.assert_check:
            assert not ans.isnan().any()
            assert not den.isnan().any()
            self.debug_vars['nonzero'] = nonzero

        return torch.cat([ans, den], dim=-1)

    def forward(self, line_o, line_d, sample_per_box=0):
        if not self.editting and not self.fixing_norm_mesh_dir:
            assert not self.enable_space_transform, "Space transform is not supported in non-editting mode"

        if not self.fixing_norm_mesh_dir:
            assert self.fixed_norm_mesh_dir, "norm mesh dir is not fixed"

        num_pts = self.pts.shape[0]
        assert self.pts_embed_raw.shape[0] == num_pts
        assert self.pts_conf_raw.shape[0] == num_pts
        assert self.pts_norm_mesh.shape[0] == num_pts
        if self.enable_space_transform:
            assert self.pts_space_transform.shape[0] == num_pts

        # force zero grad
        self.zero_grad()

        self.reg_loss = 0.0
        self.line_intersect = torch.ones(line_o.shape[0], dtype=torch.bool, device='cuda')

        pts_embed = self.pts_embed()
        pts_conf = self.pts_conf()
        line_d = line_d / line_d.norm(dim=-1, keepdim=True)

        self.interpolator_loss = []
        self.sample_pts = []
        if self.training and self.train_single_tree:
            k = random.randint(0, self.num_trees - 1)
            tree = self.trees[k]
            if self.visualize_mode:
                print(f"Using tree #{k}")
            ans = self.render(tree, line_o, line_d, pts_embed, pts_conf, sample_per_box=sample_per_box)
        else:
            ans = [self.render(tree, line_o, line_d, pts_embed, pts_conf, sample_per_box=sample_per_box, deployed_tree=deployed_tree) for tree, deployed_tree in zip(self.trees, self.deployed_trees)]
            ans = sum(ans) / self.num_trees

        if sample_per_box > 0 and len(self.sample_pts) > 0:
            mix_embed_dim = (self.embed_dim + 9 if self.enable_space_transform else self.embed_dim)
            sample_line, sample_pts, sample_emb, sample_den, sample_dis = torch.cat(self.sample_pts, dim=0).split([1, 3, mix_embed_dim, self.density_dim, 1], dim=-1)
            sample_line = (sample_line + 0.5).long().squeeze(-1)
            sample_den = sample_den.squeeze(-1)
            sample_dis = sample_dis.squeeze(-1)
            sample = (sample_line, sample_pts, sample_emb, sample_den, sample_dis)
            self.sample = sample
        else:
            sample = (None, None, None, None, None) 

        pts_conf_fixed = pts_conf.clamp(min=1e-5, max=1-1e-5)
        self.conf_reg_loss = softhardclamp(((pts_conf_fixed).log() + (1 - pts_conf_fixed).log()) + 2.40795, min=0, max=100000).mean() 
        
        # int_loss = sum([a[0] for a in self.interpolator_loss]) / max(1, sum(a[1] for a in self.interpolator_loss))
        # eloss = int_loss + conf_loss
        eloss = torch.tensor(0.0).cuda()
        if self.training:
            
            if not self.fixing_norm_mesh_dir:
                eloss = eloss + 1 * self.reg_loss / self.line_intersect.sum().clamp(min=1)

                if not self.training_st:
                    assert self.fixed_norm_mesh_dir
                    mask = torch.arange(self.pts.shape[0], device='cuda')[:65536]
                    batch = self.cur_pts_embed[mask]
                    batch_conf = self.cur_pts_conf[mask]
                    norm_mesh = self.pts_norm_mesh[mask]

                    norm_pred = self.feed_pts_norm(self.feed_fea(batch))
                    if self.enable_space_transform:
                        st = self.pts_space_transform[mask]
                        st = mat_normalize(st.reshape(*st.shape[:-1], 3, 3))
                        norm_pred = ref.l2_normalize(norm_pred.unsqueeze(-2).matmul(st.transpose(-1, -2)).squeeze(-2))

                    coef = min(self.mesh_reg_weight, max(1, self.mesh_reg_weight * self.warmup_coef / self.warmup_mult))
                    # eloss = eloss + coef * ((1 - (norm_mesh * norm_pred).sum(dim=-1).pow(2)) * batch_conf).mean(dim=-1)
                    eloss = eloss + coef * ((1 - (norm_mesh * norm_pred).sum(dim=-1)) * batch_conf).mean(dim=-1)
                else:
                    assert self.editting
                    batch = self.cur_pts_embed
                    batch_conf = self.cur_pts_conf
                    st = mat_normalize(self.pts_space_transform.reshape(-1, 3, 3))

                    norm_orig = self.orig_pts_norm_mesh
                    norm_rest = ref.l2_normalize(self.pts_norm_mesh.unsqueeze(-2).matmul(st).squeeze(-2))
                    # assume: norm_orig = norm_mesh @ st

                    coef = self.mesh_reg_weight # min(self.mesh_reg_weight, max(1, self.mesh_reg_weight * self.warmup_coef))
                    eloss = eloss + coef * ((1 - (norm_orig * norm_rest).sum(dim=-1)) * batch_conf).mean(dim=-1)



                if self.enable_space_transform:
                    st = mat_normalize(self.pts_space_transform.reshape(-1, 3, 3))
                    st_norm = (st.matmul(st.transpose(1, 2)) - torch.eye(3, device='cuda')).pow(2).mean()
                    eloss = eloss + 10 * st_norm

            else:
                eloss = self.reg_loss


        ans, den = ans.split([self.field_dim, self.density_dim], dim=-1)
        self.field = ans
        self.density = den

        ans_merged = ans * den + self.default_field_val * (1 - den)

        if self.model_background and (self.training or self.growing or self.model_background_force_render) and (
            (not self.depth_map_mode) and (not self.fixing_norm_mesh_dir) and (not self.training_st)):
            # self.posemb_bg = PosEmbed(16)
            # self.feed_bg_bottleneck = nn.Sequential(
            #     nn.Linear(self.posemb_bg.dim, 128),
            #     nn.ELU(),
            #     nn.Linear(128, 256),
            #     Res(nn.Sequential(
            #         nn.ELU(),
            #         nn.Linear(256, 256),
            #         nn.ELU(),
            #         nn.Linear(256, 256),
            #     )),
            #     nn.ELU(),
            #     nn.Linear(256, mlp_shapes[2]),
            # )
            # self.dir_enc_bg = ref.generate_dir_enc_fn(sh_deg)
            # ref_sh_dim_bg = [0, 4, 10, 20, 38, 72][sh_deg]
            # self.feed_rgb_bg = nn.Sequential(
            #     nn.Linear(mlp_shapes[2] + ref_sh_dim_bg, mlp_shapes[1]),
            #     nn.ELU(),
            #     layer2_mlp(field_dim)
            # )
            R = self.model_background_R

            # find the intersection of line_o + t*line_d and circle with radius R
            # line_o, line_d: (B, 3)
            coef = dotprod(-line_o, line_d)
            # (B)
            proj_o = line_o + coef.unsqueeze(-1) * line_d
            # (B, 3)
            
            dist = (R**2 - dotprod(proj_o, proj_o)).clamp(min=1e-9).sqrt()
            # (B)
            # the intersection point is:
            bgpt = proj_o + dist.unsqueeze(-1) * line_d

            assert (bgpt.norm(dim=-1) - R).abs().max() < 1e-3
            
            bgpt = ref.l2_normalize(bgpt)

            posemb = self.posemb_bg(bgpt)
            bottleneck = self.feed_bg_bottleneck(posemb)
            dir_enc = self.dir_enc_bg(bgpt)
            background = self.feed_rgb_bg(torch.cat([bottleneck, dir_enc], dim=-1)).sigmoid() * (1 + 0.002) - 0.001 
            # (B, 3) 
            background_den = to_density(self.feed_density_bg(bottleneck))
            # (B, 1)

            ans = ans * den + background * (1 - den) * background_den + self.default_field_val * (1 - den) * (1 - background_den)
            eloss = eloss + self.model_background_reg_weight * ((1 - den) * background_den)[self.line_intersect].mean()
        else:
            ans = ans_merged

        if self.depth_map_mode:
            ans = self.field

        if self.assert_check:
            assert not ans.isnan().any()
            assert not den.isnan().any()
            if self.training:
                assert not self.conf_reg_loss.isnan().any()
                assert not eloss.isnan().any()

        if self.render_record_mode == 2:
            # clear
            self.render_record_mode = -1
            del self.render_record
            self.render_record = [dict(), dict()]

        return ans, eloss, sample

    def deploy(self):
        self.eval()
        self.warning_emitted = False
        # with torch.no_grad():
        #     pts_embed = self.pts_embed()
        #     pts_conf = self.pts_conf()

        #     for tree_i, tree in enumerate(self.trees):
        #         (_, _, _, tree_support, _, _) = tree
        #         deployed_tree = []
        #         # logging.info(f"Deploying tree #{tree_i}")
        #         for i, support in enumerate(tree_support):
        #             if self.num_layer - self.sample_height_max <= i <= self.num_layer - self.int_height_min:
        #                 deployed_tree.append(self.fit_fea_weight(self.pts[support], pts_embed[support], pts_conf[support]))
        #             else:
        #                 deployed_tree.append(None)

        #         self.deployed_trees[tree_i] = deployed_tree

    def undeploy(self):
        self.train()
        # for a in self.deployed_trees:
        #     del a
        # del self.deployed_trees
        # self.deployed_trees = [None] * self.num_trees
        self.warning_emitted = False
            
    def pts_grow_candidate(self, sample=None, weight=None, den_thres=0.4, dis_thres=1.75, lim=1000, with_embed=False, return_ind=False):
        if sample is None:
            sample = self.sample

        dis_thres *= (self.pts.shape[0] / self.max_pts_lim) ** 0.5

        pts, emb, den, dis = sample
        ind = torch.arange(pts.shape[0], device='cuda')
        mask = (den > den_thres) & (dis > dis_thres)
        pts = pts[mask]
        emb = emb[mask]
        den = den[mask]
        dis = dis[mask]
        ind = ind[mask]

        if lim != -1 and pts.shape[0] > lim:
            if weight is None:
                weight = den
            else:
                weight = weight[mask]

            _, mask = weight.topk(lim, largest=True, dim=-1)
            pts = pts[mask]
            emb = emb[mask]
            ind = ind[mask]
            # den = den[mask]
            # dis = dis[mask]
            # weight = weight[mask]

        if return_ind:
            return ind

        if not with_embed:
            return pts
        return pts, emb

    def pts_grow_select(self, pts, weight=None, min_margin=1.75, num_trees=1, return_mask=False):
        min_margin *= (self.pts.shape[0] / self.max_pts_lim) ** 0.5
        min_margin *= self.unit_len
        pts = pts.float()
        if weight is not None:
            weight = weight.float().reshape(-1)
        preserve = torch.zeros(pts.shape[0], dtype=torch.bool, device='cuda')

        for _ in range(num_trees):
            (_, tree_lay, _, _, tree_box_d, tree_box_u) = self.arrange(pts, max_pts=0, bnd_pad_fact=0, no_logging=True, no_support=True)
            unused = torch.tensor([True], device='cuda')

            for i, (lay, box_d, box_u) in enumerate(zip(tree_lay, tree_box_d, tree_box_u)):
                satisfy = ((box_u - box_d).norm(dim=-1) < min_margin) & unused
                lay = lay[satisfy]

                # print(f"select #satisfy = {satisfy.sum()} #unused = {unused.sum()}")

                if lay.shape[0] > 0:
                    if weight is None:
                        cho = torch.randint(low=0, high=lay.shape[1], size=[lay.shape[0]], device='cuda')
                    else:
                        cho = weight[lay].argmax(dim=-1)

                    cho = lay.gather(1, cho[:, None]).squeeze(-1)
                    preserve[cho] = True

                unused[satisfy] = False
                if unused.sum() == 0:
                    break
                unused = torch.cat([unused, unused], dim=0)
        if return_mask:
            return preserve
        return pts[preserve]

    def pts_prune(self, lim=None, conf_thres=0.1):
        if lim is None:
            lim = self.pts.shape[0] // 4
        conf = self.pts_conf()

        removed = conf <= conf_thres

        if removed.sum() > lim:
            _, ind = conf.topk(lim, largest=False, dim=0)
            removed[:] = False
            removed[ind] = True

        assert removed.sum().item() <= lim
        return removed   

    def pseudo_prune(self, prune_mask=None):
        if prune_mask is None:
            return
        logging.info(f"pseudo-prune {prune_mask.sum().item()} points") 

        # prune without regenerating parameters and trees
        with torch.no_grad():
            new_pts_conf_raw = self.pts_conf_raw.clone()
            new_pts_conf_raw[prune_mask] = -1000
            self.pts_conf_raw.set_(new_pts_conf_raw)

    def maintain(self, prune_mask=None, prune_outlier_fn=None, grow_pts=None, grow_pts_embed=None, grow_pts_conf=None, grow_pts_weight=None, remove_outliers=True):
        self.eval()

        with torch.no_grad():
            if prune_mask is not None:
                reserve = ~prune_mask
                self.pts = self.pts[reserve]
                self.pts_embed_raw = nn.Parameter(self.pts_embed_raw[reserve])
                self.pts_conf_raw = nn.Parameter(self.pts_conf_raw[reserve])
                if self.enable_space_transform:
                    self.pts_space_transform = nn.Parameter(self.pts_space_transform[reserve])
                logging.info(f"maintain: prune {prune_mask.sum().item()} points")

            if grow_pts is not None:
                if self.enable_space_transform:
                    assert grow_pts_embed is not None
                    grow_pts_embed, grow_pts_st = grow_pts_embed.split([self.embed_dim, 9], dim=-1)

                if grow_pts_embed is None:
                    grow_pts_embed = torch.randn(grow_pts.shape[0], self.embed_dim, device='cuda')          
                else:
                    grow_pts_embed = (grow_pts_embed - self.pts_embed_norm.bias) / denominator(self.pts_embed_norm.weight, eps=1e-6)
                if grow_pts_conf is None:
                    grow_pts_conf = torch.zeros(grow_pts.shape[0], dtype=torch.float, device='cuda')
                else:
                    grow_pts_conf = grow_pts_conf.clamp(min=1e-6, max=1-1e-6).logit()
                
                # This scaling should be changed with query_pts_embed
                grow_pts_embed /= 2 * grow_pts_conf.sigmoid().unsqueeze(-1)

                grow_pts_embed = grow_pts_embed * (self.pts_embed_norm.running_var + self.pts_embed_norm.eps).pow(0.5) + self.pts_embed_norm.running_mean
                if not self.disable_conf_recenter:
                    grow_pts_conf += self.pts_conf_norm.mean

                if self.pts.shape[0] + grow_pts.shape[0] > self.max_pts_lim:
                    logging.info(f"maintain: grown exceed max_pts {self.pts.shape[0] + grow_pts.shape[0]} > {self.max_pts_lim}")
                    if grow_pts_weight is None:
                        mask = torch.randint(low=0, high=grow_pts.shape[0], size=[self.max_pts_lim - self.pts.shape[0]], device='cuda')
                    else:
                        mask = grow_pts_weight.reshape(-1).topk(self.max_pts_lim - self.pts.shape[0], largest=True, sorted=False, dim=0)[1]
                    logging.info(f"maintain: preserve {mask.shape[0]} points")
                    grow_pts = grow_pts[mask]
                    grow_pts_embed = grow_pts_embed[mask]
                    grow_pts_conf = grow_pts_conf[mask]
                    if self.enable_space_transform:
                        grow_pts_st = grow_pts_st[mask]

                self.pts = torch.cat([self.pts, grow_pts.cuda()], dim=0)
                self.pts_embed_raw = nn.Parameter(torch.cat([self.pts_embed_raw, grow_pts_embed], dim=0))
                self.pts_conf_raw = nn.Parameter(torch.cat([self.pts_conf_raw, grow_pts_conf], dim=0))
                if self.enable_space_transform:
                    self.pts_space_transform = nn.Parameter(torch.cat([self.pts_space_transform, grow_pts_st], dim=0))

                logging.info(f"maintain: grow {grow_pts.shape[0]} points")

            if prune_outlier_fn:
                ind = torch.arange(self.pts.shape[0], device='cuda')
                
                for ofn in prune_outlier_fn:
                    mask = ofn(self.pts)
                    ind = ind[mask]
                    self.pts = self.pts[mask]

                logging.info(f"maintain: remove {self.pts_embed_raw.shape[0] - ind.shape[0]} image outliers")
                self.pts_embed_raw = nn.Parameter(self.pts_embed_raw[ind])
                self.pts_conf_raw = nn.Parameter(self.pts_conf_raw[ind])
                if self.enable_space_transform:
                    self.pts_space_transform = nn.Parameter(self.pts_space_transform[ind])
                


            if remove_outliers:
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(self.pts.cpu().numpy())
                cloud, mask = cloud.remove_statistical_outlier(nb_neighbors=64, std_ratio=4.5)
                mask = torch.tensor(np.asarray(mask)).cuda()

                logging.info(f"maintain: remove {self.pts.shape[0] - mask.shape[0]} outliers")
            
                self.pts = self.pts[mask]
                self.pts_embed_raw = nn.Parameter(self.pts_embed_raw[mask])
                self.pts_conf_raw = nn.Parameter(self.pts_conf_raw[mask])
                if self.enable_space_transform:
                    self.pts_space_transform = nn.Parameter(self.pts_space_transform[mask])
                
        
        self.rebuild_trees()
        self.train()

    def fix_norm_mesh_dir(self, train_loader):
        if self.fixed_norm_mesh_dir:
            logging.info("fix_norm_mesh_dir: already fixed")
            return

        # assert self.editting, "model is not in scene editting mode"
        from tqdm.contrib.logging import logging_redirect_tqdm
        if not self.norm_mesh_complete:
            logging.info("fix_norm_mesh_dir: norm mesh is not complete")
            self.calculate_norm_mesh(remove_outlier=False)

        self.undeploy()
        # self.enable_st()
        self.training_st = True
        self.fixing_norm_mesh_dir = True
        params = list(self.parameters())
        for p in params:
            p.requires_grad_(False)

        pts_norm_mesh = self.pts_norm_mesh.clone()

        sign_coef = torch.zeros(self.pts.shape[0], device='cuda').requires_grad_(True)
        opt = torch.optim.SGD([sign_coef], lr=1)

        tbar = tqdm(train_loader, mininterval=5)
        tbar.set_description(f"Fix dir")

        for index, (line_o, line_d, _) in enumerate(tbar):
            line_o = line_o.cuda()
            line_d = line_d.cuda()
            
            sign_coef_dth = sign_coef.detach()
            placeholder = torch.ones_like(sign_coef) + (sign_coef - sign_coef_dth)
            # just take the sign_coef's grad, no optimization
            self.pts_norm_mesh = pts_norm_mesh * placeholder.unsqueeze(-1)

            assert self.pts_norm_mesh.requires_grad

            with logging_redirect_tqdm():
                _, eloss, _ = self(line_o, line_d, sample_per_box=0)

            eloss = eloss / 10000

            assert not eloss.isnan().any()
            opt.zero_grad()
            eloss.backward()
            opt.step()
            self.warmup_step(len(train_loader))

            if index % 25 == 0:
                pos = (sign_coef > 0).sum().item() / sign_coef.shape[0] * 100
                neg = (sign_coef < 0).sum().item() / sign_coef.shape[0] * 100
                zero = (sign_coef == 0).sum().item() / sign_coef.shape[0] * 100
                tbar.set_postfix_str(f"+: {pos:.1f}%, -: {neg:.1f}%, 0: {zero:.1f}%, eloss: {eloss:.4f}")

        with torch.no_grad():
            self.pts_norm_mesh = pts_norm_mesh * sign_coef.detach().sgn().unsqueeze(-1)

        for p in params:
            p.requires_grad_(True)
        self.training_st = False
        self.fixing_norm_mesh_dir = False
        self.fixed_norm_mesh_dir = True
        self.deploy() 


    def estimate_initial_st(self, num_nn=64, batch_size=65536, num_iter=32, num_smooth_iter=4):
        from sklearn.neighbors import NearestNeighbors
        from tqdm import tqdm
        import gc
        if not self.editting:
            logging.info("estimate_initial_st: not in editting mode, mimicing")
            self.orig_pts = self.pts.clone()
            self.orig_pts_norm_mesh = self.pts_norm_mesh.clone()
            self.orig_unit_len = self.unit_len

        assert self.fixed_norm_mesh_dir, "norm mesh dir is not fixed"
        assert self.norm_mesh_complete, "norm mesh is not complete"        
        assert self.pts.shape[0] == self.orig_pts.shape[0], "pts and orig_pts should have the same number of points"

        if not self.init_space_transform:
            logging.info("estimate_initial_st: no init st")
            self.pts_space_transform.requires_grad_(False)
            return

        if self.naive_plot_mode:
            logging.info("estimate_initial_st: naive plot mode, skip")
            return

        pts_conf = self.pts_conf().clamp(min=1e-5)
        pts_np = self.pts.cpu().numpy()
        orig_pts_np = self.orig_pts.cpu().numpy()
        knn_solver = NearestNeighbors().fit(pts_np)
        orig_knn_solver = NearestNeighbors().fit(orig_pts_np)

        def calc():
            M_candidates = []
            knn_buffer = []
            for i in tqdm(range(0, self.pts.shape[0], batch_size)):
                dist, ind = knn_solver.kneighbors(pts_np[i:i+batch_size], n_neighbors=num_nn)
                dist = torch.tensor(dist).cuda()
                ind = torch.tensor(ind).cuda()
                # (N, num_nn)

                orig_dist, _ = orig_knn_solver.kneighbors(orig_pts_np[i:i+batch_size], n_neighbors=num_nn)
                orig_dist_lim = torch.tensor(orig_dist).amax(dim=-1, keepdim=True).cuda() * 2
                # (N, 1)
                orig_dist = (self.orig_pts[i:i+batch_size].unsqueeze(1) - self.orig_pts[ind]).norm(dim=-1)
                # (N, num_nn)

                knn_buffer.append((dist.cpu(), ind.cpu(), orig_dist.cpu(), orig_dist_lim.cpu()))

            for it in range(num_iter):           
                M_list = []

                for i in range(0, self.pts.shape[0], batch_size):
                                
                    dist, ind, orig_dist, orig_dist_lim = knn_buffer[i // batch_size]
                    dist = dist.cuda()
                    ind = ind.cuda()
                    orig_dist = orig_dist.cuda()
                    orig_dist_lim = orig_dist_lim.cuda()

                    value = 1e7 * (orig_dist > orig_dist_lim)
                    value += 1e7 * (dist < 4 * self.unit_len)
                    value += 1e7 * (orig_dist < 4 * self.orig_unit_len)
                    value += 2 / pts_conf[ind]
                    value += dist / (self.unit_len * 4) / pts_conf[ind]
                    value += orig_dist / (self.orig_unit_len * 4) / pts_conf[ind]

                    nn_vec = ref.l2_normalize(self.pts[ind] - self.pts[i:i+batch_size].unsqueeze(1))
                    norm_crossprod = nn_vec.cross(self.pts_norm_mesh[i:i+batch_size].unsqueeze(1)).norm(dim=-1)
                    orig_nn_vec = ref.l2_normalize(self.orig_pts[ind] - self.orig_pts[i:i+batch_size].unsqueeze(1))
                    orig_norm_crossprod = orig_nn_vec.cross(self.orig_pts_norm_mesh[i:i+batch_size].unsqueeze(1)).norm(dim=-1)
                    value += 10 / norm_crossprod.abs().clamp(min=1e-5)
                    value += 10 / orig_norm_crossprod.abs().clamp(min=1e-5)
                    randmov = torch.randn_like(value) * value[value < 1e7/2].std(dim=-1, keepdim=True) * 4
                    cho1 = (value + randmov).argmin(dim=-1)

                    cho1_nn_vec = nn_vec.gather(1, cho1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)).squeeze(1)
                    cho1_tripleprod = (nn_vec.cross(cho1_nn_vec.unsqueeze(1)) * self.pts_norm_mesh[i:i+batch_size].unsqueeze(1)).sum(dim=-1)
                    orig_cho1_nn_vec = orig_nn_vec.gather(1, cho1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)).squeeze(1)
                    orig_cho1_tripleprod = (orig_nn_vec.cross(orig_cho1_nn_vec.unsqueeze(1)) * self.orig_pts_norm_mesh[i:i+batch_size].unsqueeze(1)).sum(dim=-1)
                    value += 10 / cho1_tripleprod.abs().clamp(min=1e-5)
                    value += 10 / orig_cho1_tripleprod.abs().clamp(min=1e-5)
                    randmov = torch.randn_like(value) * value[value < 1e7/2].std(dim=-1, keepdim=True) * 4
                    cho2 = (value + randmov).argmin(dim=-1)

                    axis = []
                    orig_axis = []

                    for cho in [cho1, cho2]:
                        axis.append(nn_vec.gather(1, cho[:, None, None].expand(-1, -1, 3)).squeeze(1))
                        orig_axis.append(orig_nn_vec.gather(1, cho[:, None, None].expand(-1, -1, 3)).squeeze(1))

                    x_axis = ref.l2_normalize(self.pts_norm_mesh[i:i+batch_size])
                    y_axis = axis[0]
                    y_axis = ref.l2_normalize(y_axis - x_axis * (x_axis * y_axis).sum(dim=-1, keepdim=True))
                    z_axis = axis[1]
                    z_axis = ref.l2_normalize(z_axis - x_axis * (x_axis * z_axis).sum(dim=-1, keepdim=True))
                    z_axis = ref.l2_normalize(z_axis - y_axis * (y_axis * z_axis).sum(dim=-1, keepdim=True))
                    system = torch.stack([x_axis, y_axis, z_axis], dim=-2)
                    # (N, 3, 3)

                    orig_x_axis = ref.l2_normalize(self.orig_pts_norm_mesh[i:i+batch_size])
                    orig_y_axis = orig_axis[0]
                    orig_y_axis = ref.l2_normalize(orig_y_axis - orig_x_axis * (orig_x_axis * orig_y_axis).sum(dim=-1, keepdim=True))
                    orig_z_axis = orig_axis[1]
                    orig_z_axis = ref.l2_normalize(orig_z_axis - orig_x_axis * (orig_x_axis * orig_z_axis).sum(dim=-1, keepdim=True))
                    orig_z_axis = ref.l2_normalize(orig_z_axis - orig_y_axis * (orig_y_axis * orig_z_axis).sum(dim=-1, keepdim=True))
                    orig_system = torch.stack([orig_x_axis, orig_y_axis, orig_z_axis], dim=-2)
                    # (N, 3, 3)

                    # find a matrix M, orig_system = system @ M
                    # M = system^-1 @ orig_system
                    M = system.transpose(-1, -2) @ orig_system
                    # (N, 3, 3)

                    M = M.view(-1, 9)
                    # (N, 9)

                    M_list.append(M)
                
                M = torch.cat(M_list, dim=0)
                
                st = M.reshape(-1, 3, 3)
                norm_orig = self.orig_pts_norm_mesh
                norm_rest = ref.l2_normalize(self.pts_norm_mesh.unsqueeze(-2).matmul(st).squeeze(-2))
                error = 1 - (norm_orig * norm_rest).sum(dim=-1)
                self.initial_st_error = error

                error_mask = error > 1e-5

                logging.info(f"Initial ST #{it} mean_error = {error.mean().item()} #error_mask = {error_mask.sum().item()}")

                if error_mask.sum().item() > 0:
                    # do interpolation with knn
                    dist, ind = knn_solver.kneighbors(self.pts[error_mask].cpu().numpy(), n_neighbors=num_nn)
                    dist = torch.tensor(dist).cuda()
                    ind = torch.tensor(ind).cuda()

                    coef = pts_conf[ind].log() - (dist + self.unit_eps()).log()
                    coef[error_mask[ind]] = -1e9
                    coef = coef.softmax(dim=-1)

                    M[error_mask] = mat_normalize((M[ind] * coef[:, :, None]).sum(dim=1).reshape(-1, 3, 3)).float().reshape(-1, 9)
                
                M_candidates.append(M)

            M = mat_normalize(sum(M_candidates).reshape(-1, 3, 3)).reshape(-1, 9)
            del M_candidates
            del knn_buffer
            gc.collect()
            return M

        M = calc()

        batch_size //= 8
        coef_buffer = []
        for i in tqdm(range(0, self.pts.shape[0], batch_size)):
            dist, ind = knn_solver.kneighbors(pts_np[i:i+batch_size], n_neighbors=num_nn // 4)
            dist = torch.tensor(dist).float().cuda()
            ind = torch.tensor(ind).cuda()
            coef = pts_conf[ind].log() - dist.clamp(min=self.unit_len * 4).log()
            coef = coef.softmax(dim=-1).unsqueeze(-1)
            coef_buffer.append((coef.cpu(), ind.cpu()))

        for it in tqdm(range(num_smooth_iter)):
            # make smooth
            def smooth_step(M):
                M1 = M.cpu()
                for i in range(0, self.pts.shape[0], batch_size):
                    coef, ind = coef_buffer[i // batch_size]
                    M[i:i+batch_size] = mat_normalize(M1[ind].cuda().mul_(coef.cuda()).sum(dim=1).reshape(-1, 3, 3)).float().reshape(-1, 9)
                del M1
                gc.collect()
                return M

            M = smooth_step(M)

        if self.enable_space_transform:
            with torch.no_grad():
                self.pts_space_transform.set_(M)

        return M
        

    def train_st(self, train_loader, num_epoch=0, fix_loader=None, estimate_initial_st=True):
        assert self.editting, "model is not in scene editting mode"
        if fix_loader is None:
            fix_loader = train_loader

        self.enable_st()
        self.fix_norm_mesh_dir(fix_loader)
        if estimate_initial_st:
            self.estimate_initial_st()

        if not self.init_space_transform:
            logging.info("no initial space transform, skip training")
            self.pts_space_transform.requires_grad_(False)
            return

        if self.naive_plot_mode:
            logging.info("naive plot mode, skip training ST")
            return

        from tqdm.contrib.logging import logging_redirect_tqdm
        self.undeploy()
        
        self.training_st = True

        params = list(self.parameters())
        for p in params:
            p.requires_grad_(False)
        self.pts_space_transform.requires_grad_(True)

        self.warmup_coef = 1
        opt = torch.optim.Adam([self.pts_space_transform], lr=5e-4)

        for epoch in range(1, num_epoch + 1):
            tbar = tqdm(train_loader, mininterval=5)
            tbar.set_description(f"Train ST #{epoch}")
            for index, (line_o, line_d, _) in enumerate(tbar):
                line_o = line_o.cuda()
                line_d = line_d.cuda()

                with logging_redirect_tqdm():
                    _, eloss, _ = self(line_o, line_d, sample_per_box=0)

                assert not eloss.isnan().any()
                opt.zero_grad()
                eloss.backward()
                opt.step()
                self.warmup_step(len(train_loader))

                if index % 10 == 0:
                    tbar.set_postfix_str(f"eloss: {eloss:.4f}")

        for p in params:
            p.requires_grad_(True)
        self.training_st = False
        self.deploy()
        
    def prune_and_grow(self, train_loader, prune_outlier_fn):
        from tqdm.contrib.logging import logging_redirect_tqdm
        self.deploy()
        self.growing = True
        with torch.no_grad():
            tbar = tqdm(train_loader, mininterval=5)
            tbar.set_description(f"P&G")

            samples = []

            for index, (line_o, line_d, target) in enumerate(tbar):
                line_o = line_o.cuda()
                line_d = line_d.cuda()
                target = target.cuda()
                target, density_target = target.split([3, 1], dim=-1)

                background = target.min(dim=-1).values < -0.5
                # target[background] += 4
                target = target[~background]
                line_o = line_o[~background]
                line_d = line_d[~background]
                density_target = density_target[~background]

                with logging_redirect_tqdm():
                    pred_img, _, sample = self(line_o, line_d, sample_per_box=16)
                    predict = self.field
                    density_predict = self.density
                
                if sample[0].shape[0] > 0:
                    with logging_redirect_tqdm():
                        # ins = self.line_intersect
                        (sample_line, sample_pts, sample_emb, sample_den, sample_dis) = sample
                        
                        if not self.regrow_mode:
                            dis_thres = 1.75
                            dis_thres_fixed = dis_thres * (self.pts.shape[0] / self.max_pts_lim) ** 0.5
                            mask = sample_dis > dis_thres_fixed
                            sample_line = sample_line[mask]
                            sample_pts = sample_pts[mask]
                            sample_emb = sample_emb[mask]
                            sample_den = sample_den[mask]
                            sample_dis = sample_dis[mask]
                        else:
                            dis_thres = 0.0

                        max_density = torch.zeros_like(line_o[:, 0])
                        # prevent scatter_reduce from printing
                        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                            max_density.scatter_reduce_(0, sample_line.reshape(-1), sample_den.reshape(-1), reduce='amax')

                        mask = sample_den >= max_density[sample_line]
                        sample_line = sample_line[mask]
                        sample_pts = sample_pts[mask]
                        sample_emb = sample_emb[mask]
                        sample_den = sample_den[mask]
                        sample_dis = sample_dis[mask]

                        sample_den /= density_target[sample_line].reshape(sample_den.shape).clamp(min=1e-5)
                        sample_den.clamp_(max=1.0)

                        targ_img = target * density_target + self.default_field_val * (1 - density_target)
                        img_loss = (pred_img - targ_img).norm(dim=-1).reshape(-1)

                        if self.grow_img_loss:
                            ploss = img_loss
                        else:
                            field_loss = (predict - target.clamp(min=0, max=1)).pow(2).mean(dim=-1)
                            density_loss = (density_predict - density_target.clamp(min=0, max=1)).pow(2).reshape(-1)
                            field_loss[~self.line_intersect] = 1.0
                            ploss = img_loss + field_loss * density_target.reshape(-1) + density_loss * (1 - density_target.reshape(-1)) + 0.1 * density_loss


                        sample_dif = ploss[sample_line].reshape(-1)
                        # sample_dif *= density_target[sample_line].reshape(-1)

                        sample = (sample_pts, sample_emb, sample_den, sample_dis)
                        base_lim = (108 if self.max_pts < self.max_pts_lim else 96)

                        if self.regrow_mode:
                            # in regrow mode, use allow more points, and sample with higher density
                            base_lim *= 2
                            sample_dif = sample_den + 0.1 * sample_dif

                        mask = self.pts_grow_candidate(sample, weight=sample_dif, lim=int(base_lim * self.max_pts_lim / self.pts.shape[0] + 0.5), dis_thres=dis_thres, return_ind=True)
                        sample_pts = sample_pts[mask]
                        sample_emb = sample_emb[mask]
                        sample_den = sample_den[mask]
                        sample_dif = sample_dif[mask]
                    
                        if sample_pts.shape[0] > 0:
                            samples.append(torch.cat([sample_pts, sample_emb, sample_den.reshape(-1, 1), sample_dif.reshape(-1, 1)], dim=-1).float())

                if index % 10 == 0:
                    num_samples = sum([s.shape[0] for s in samples])
                    tbar.set_postfix_str(f"samples: {num_samples}")

                if index % 100 == 0 and index > 0:
                    if len(samples) > 0:
                        mix_embed_dim = (self.embed_dim + 9 if self.enable_space_transform else self.embed_dim)
                        sample_pts, sample_emb, sample_den, sample_dif = torch.cat(samples, dim=0).split([3, mix_embed_dim, 1, 1], dim=-1)

                        mask = self.pts_grow_select(sample_pts, weight=sample_dif, min_margin=2.25, num_trees=1, return_mask=True)
                        sample_pts = sample_pts[mask]
                        sample_emb = sample_emb[mask]
                        sample_den = sample_den[mask]
                        sample_dif = sample_dif[mask]

                        if index % 500 == 0:
                            with logging_redirect_tqdm():
                                logging.info(f"P&G {index}/{len(tbar)} samples: {sample_pts.shape[0]}")

                        samples = []
                        samples.append(torch.cat([sample_pts, sample_emb, sample_den.reshape(-1, 1), sample_dif.reshape(-1, 1)], dim=-1).float())

            if len(samples) > 0:
                mix_embed_dim = (self.embed_dim + 9 if self.enable_space_transform else self.embed_dim)
                sample_pts, sample_emb, sample_den, sample_dif = torch.cat(samples, dim=0).split([3, mix_embed_dim, 1, 1], dim=-1)

                mask = self.pts_grow_select(sample_pts, weight=sample_dif, min_margin=2.75, num_trees=1, return_mask=True)
                grow_pts = sample_pts[mask]
                grow_emb = sample_emb[mask]
                grow_dif = sample_dif[mask]

                perm = torch.randperm(grow_pts.shape[0], device='cuda')
                grow_pts = grow_pts[perm]
                grow_emb = grow_emb[perm]
                grow_dif = grow_dif[perm]
            else:
                grow_pts = None
                grow_emb = None
                grow_dif = None
            
            prune_lim = None
            if grow_pts is not None:
                prune_lim = grow_pts.shape[0] * 9 // 8
            prune_mask = self.pts_prune(lim=prune_lim)

            if self.aggressive_prune and (grow_pts is not None):
                if self.pts.shape[0] > self.max_pts_lim // 2:
                    if prune_mask.sum().item() + 1 < grow_pts.shape[0]:
                        # force to prune the first several points
                        num_prune = (grow_pts.shape[0] - prune_mask.sum().item()) // 4
                        num_prune = min(num_prune, self.pts.shape[0] // 16)
                        logging.info(f"force to prune first {num_prune} points")
                        prune_mask[:num_prune] = True

            if self.regrow_mode:
                prune_mask |= True

            if prune_mask.sum().item() > 0:
                prune_pts = self.pts[prune_mask]
            else:
                prune_mask = None
                prune_pts = None

                
            self.maintain(prune_mask=prune_mask, prune_outlier_fn=prune_outlier_fn, 
                          grow_pts=grow_pts, grow_pts_embed=grow_emb, grow_pts_weight=grow_dif)       

        self.undeploy()
        self.growing = False
        return grow_pts, prune_pts, self.pts.detach(), self.pts_norm_mesh.detach()
