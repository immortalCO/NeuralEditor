import torch
from scipy import special
import math

# Translating https://github.com/google-research/multinerf/blob/main/internal/ref_utils.py to pytorch

def reflect(viewdirs, normals):
    return 2 * torch.sum(normals * viewdirs, dim=-1, keepdim=True) * normals - viewdirs

def l2_normalize(x, eps=1e-5, return_norm=False):
    norm = torch.norm(x, dim=-1, keepdim=True)
    y = x / norm.clamp(min=eps)

    if return_norm:
        return y, norm
    else:
        return y

def generalized_binomial_coeff(a, k):
    return (a - torch.arange(k, dtype=torch.float64, device='cuda')).prod() / math.factorial(k)

def assoc_legendre_coeff(l, m, k):
    return ((-1)**m * 2**l * math.factorial(l) 
        / (math.factorial(k) * math.factorial(l-k-m)) 
        * generalized_binomial_coeff(0.5 * (l+k+m-1), l))


def sph_harm_coeff(l, m, k):
    return (torch.sqrt(
        (2.0 * l + 1.0) * math.factorial(l - m) /
        (4.0 * math.pi * math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))


def get_ml_array(deg_view):
    ml_item = lambda l : torch.stack([torch.arange(l + 1, device='cuda'), torch.full((l + 1,), l, device='cuda')], dim=1)
    ml_list = torch.cat([ml_item(2**i) for i in range(deg_view)], dim=0).T
    return ml_list

def generate_ide_fn(deg_view):
    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view - 1)


    mat = torch.zeros((l_max + 1, ml_array.shape[1]), device='cuda')
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k).float()

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).
        Args:
        xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
        kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.
        Returns:
        An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        # vmz = jnp.concatenate([z**i for i in range(mat.shape[0])], axis=-1)
        vmz = torch.cat([z**i for i in range(mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        # vmxy = jnp.concatenate([(x + 1j * y)**m for m in ml_array[0, :]], axis=-1)
        vmxy = torch.cat([(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)


        # Get spherical harmonics.
        # sph_harms = vmxy * math.matmul(vmz, mat)
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        # ide = sph_harms * jnp.exp(-sigma * kappa_inv)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        # return jnp.concatenate([jnp.real(ide), jnp.imag(ide)], axis=-1)
        return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn


def generate_dir_enc_fn(deg_view):

    integrated_dir_enc_fn = generate_ide_fn(deg_view)

    def dir_enc_fn(xyz, unused_kappa_inv=None):
        """Function returning directional encoding (DE)."""
        # return integrated_dir_enc_fn(xyz, jnp.zeros_like(xyz[..., :1]))
        return integrated_dir_enc_fn(xyz, torch.zeros_like(xyz[..., :1]))

    return dir_enc_fn


def linear_to_srgb(linear, eps=1e-6):
    srgb0 = linear * 12.92
    srgb1 = 1.055 * torch.pow(linear.clamp(min=eps), 1.0 / 2.4) - 0.055
    return torch.where(linear <= 0.0031308, srgb0, srgb1)