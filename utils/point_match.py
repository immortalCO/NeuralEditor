import torch
from sklearn.neighbors import NearestNeighbors
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm
import open3d as o3d
import numpy as np

def plot(cloud, rgb=None):
    # placeholder
    pass

def contour_rgb(pts):
    # placeholder
    pass

def perminv(p):
    inv = torch.zeros_like(p)
    inv[p] = torch.arange(len(p))
    return inv

def match2d_raw(A, B):
    from point_match_cpp import match2d as match2d_cpp
    # placeholder
    assert A.shape[1] == 2
    assert B.shape[1] == 2
    perm = match2d_cpp(A.cpu().tolist(), B.cpu().tolist())
    return torch.tensor(perm, dtype=torch.long, device=A.device)

def match2d(A, B, fold=25):
    N = A.shape[0]
    K = (N + fold - 1) // fold

    perm = torch.zeros(N, dtype=torch.long, device='cuda')
    _, A_ind = A[:, 0].sort(descending=False)
    _, B_ind = B[:, 0].sort(descending=False)

    for i in range(0, N, K):
        a = A_ind[i : i + K]
        b = B_ind[i : i + K]
        pa = A[a]
        pb = B[b]

        p2d = match2d_raw(pa, pb)
        perm[a] = b[p2d]
    
    return perm

def match_alt(A, B, fold=200, debug=False):
    N = A.shape[0]
    K = (N + fold - 1) // fold
    logging.info(f"Matching N = {N} K = {K} fold = {fold}")

    perm = torch.zeros(N, dtype=torch.long, device='cuda')

    _, A_ind = A[:, 0].sort(descending=False)
    _, B_ind = B[:, 0].sort(descending=False)

    counter = 0
    for i in range(0, N, K):
        
        a = A_ind[i : i + K]
        b = B_ind[i : i + K]
        pa = A[a]
        pb = B[b]

        p2d = match2d(pa[:, 1:], pb[:, 1:])
        perm[a] = b[p2d]

        if counter % (fold // 10) == 0:
            with logging_redirect_tqdm():
                la, ra = pa[:, 0].aminmax()
                lb, rb = pb[:, 0].aminmax()
                dis = (pa - pb[p2d]).norm(dim=1)
                logging.info(f"Matching #{counter} xa = [{la:.4f}, {ra:.4f}] xb = [{lb:.4f}, {rb:.4f}] dis mean = {dis.mean().item():.3f} std = {dis.std().item():.3f}")

            if debug:
                plot(pa, rgb=contour_rgb(pa))
                plot(pb, rgb=contour_rgb(pb))
        counter += 1
        
    dis = (A - B[perm]).norm(dim=-1)
    return perm, dis     

def match(A, B, k=128, dim_coef=[1.02, 1.01, 1]):
    from point_match_cpp import match as match_cpp
    # _, V, _ = torch.pca_lowrank(torch.cat([A, B], dim=0))
    V = torch.cat([A, B]).std(dim=0) * torch.tensor(dim_coef).cuda()
    perm = match_cpp(k, (A / V).cpu().tolist(), (B / V).cpu().tolist())
    perm = torch.tensor(perm, dtype=torch.long, device=A.device)

    dis = (A - B[perm]).norm(dim=-1)
    return perm, dis    

MORPH = "./datasets/Morph/ns_chair_hotdog"
names = torch.load(f"{MORPH}/scene_names.pth")
info = torch.load(f"{MORPH}/info.pth")
clouds = info['points'].cuda()

index_0 = torch.tensor(info["ordered_index_0"]).cuda()
index_1 = torch.tensor(info["ordered_index_1"]).cuda()
cloud_0 = torch.load(f"{MORPH}/cloud_{names[0]}.pth").cuda()[index_0]
cloud_1 = torch.load(f"{MORPH}/cloud_{names[1]}.pth").cuda()[index_1]

device = clouds.device
N = clouds.shape[1]
assert index_0.shape[0] == N
assert index_1.shape[0] == N

_, _, M = torch.pca_lowrank(torch.cat([cloud_0, cloud_1], dim=0))
mask = torch.randperm(N)[:524288]
plot(cloud_0[mask] @ M, rgb=contour_rgb(cloud_0)[mask])
plot(cloud_1[mask] @ M, rgb=contour_rgb(cloud_1)[mask])

inliers = []
logging.info(f"Matching index_0 ~ 0")
perm, _ = match(clouds[0] @ M, cloud_0[index_0] @ M)
index_0 = index_0[perm]

for i in range(0, clouds.shape[0]):
    if i > 0:
        with logging_redirect_tqdm():
            logging.info(f"Matching {i-1} ~ {i}")

        perm, match_dis = match(clouds[i - 1] @ M, clouds[i] @ M)
        clouds[i] = clouds[i][perm]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(clouds[i].cpu().numpy())
    _, index = pcd.remove_statistical_outlier(nb_neighbors=64, std_ratio=3.5)
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[torch.from_numpy(np.array(index)).to(device)] = True
    inliers.append(mask)

    if i > 0:
        mean = match_dis.mean()
        std = match_dis.std()
        outliers = match_dis > mean + 8 * std
        inliers[i - 1][outliers] = False
        inliers[i][outliers] = False

logging.info("Matching -1 ~ index_1")
perm, _ = match(clouds[-1] @ M, cloud_1[index_1] @ M)
index_1 = index_1[perm]

index_list = []
for mask in inliers:
    index_list.append(torch.arange(N, device=device)[mask])