import torch

def A():
    return torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=torch.float, device='cuda')


def h_poly(x):
    x = x.unsqueeze(-2) ** torch.arange(4, device='cuda').unsqueeze(-1)
    return A().matmul(x)

def h_int_poly(x):
    range15 = torch.arange(1, 5, device='cuda', dtype=torch.float).unsqueeze(-1)
    x = x.unsqueeze(-2) ** range15 / range15
    return A().matmul(x)

def gather(x, qi):
    for _ in range(len(x.shape) - len(qi.shape)):
        qi = qi.unsqueeze(0)
    return x.gather(-1, qi.expand(*x.shape[:-1], -1))


def query(x, y, qx, assume_uniform_x=False):
    assert assume_uniform_x or (len(x.shape) == len(qx.shape) == 1)
    m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
    m = torch.cat([m[..., :1], (m[..., 1:] + m[..., :-1])/2, m[..., -1:]], dim=-1)

    if assume_uniform_x:
        d = (x[..., 1:] - x[..., :-1]).mean(dim=-1)
        qi = (qx / d.unsqueeze(-1)).ceil().long() - 1
    else:
        qi = torch.searchsorted(x[..., 1:].contiguous(), qx)
    qi.clamp_(min=0, max=x.shape[-1] - 2)
    qi1 = qi + 1
    qdx = gather(x, qi1) - gather(x, qi)

    qh0, qh1, qh2, qh3 = tuple(map(lambda x : x.squeeze(-2), h_poly((qx - gather(x, qi)) / qdx).split(1, dim=-2)))
    
    return qh0*gather(y,qi) + qh1*gather(m,qi)*qdx + qh2*gather(y,qi1) + qh3*gather(m,qi1)*qdx

def query_int(x, y, qx, assume_uniform_x=False):
    assert assume_uniform_x or (len(x.shape) == len(qx.shape) == 1)
    m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
    m = torch.cat([m[..., :1], (m[..., 1:] + m[..., :-1])/2, m[..., -1:]], dim=-1)

    if assume_uniform_x:
        d = (x[..., 1:] - x[..., :-1]).mean(dim=-1)
        qi = (qx / d.unsqueeze(-1)).ceil().long() - 1
    else:
        qi = torch.searchsorted(x[..., 1:].contiguous(), qx)
    qi.clamp_(min=0, max=x.shape[-1] - 2)
    qi1 = qi + 1
    qdx = gather(x, qi1) - gather(x, qi)

    Y = torch.zeros_like(y)
    Y[..., 1:] = (x[..., 1:]-x[..., :-1])*((y[..., :-1]+y[..., 1:])/2 + (m[..., :-1] - m[..., 1:])*(x[..., 1:]-x[..., :-1])/12)
    Y = Y.cumsum(dim=-1)
    qh0, qh1, qh2, qh3 = tuple(map(lambda x : x.squeeze(-2), h_int_poly((qx - gather(x, qi)) / qdx).split(1, dim=-2)))
    return gather(Y,qi) + qdx * (qh0*gather(y,qi) + qh1*gather(m,qi)*qdx + qh2*gather(y,qi1) + qh3*gather(m,qi1)*qdx)

if __name__ == "__main__":
    import matplotlib.pylab as P
    import torch as T

    x = T.linspace(0, 6, 7).cuda()
    y = x.sin()
    xs = T.linspace(0, 6, 101).cuda()
    ys = query(x, y, xs)
    Ys = query_int(x, y, xs)
    P.scatter(x, y, label='Samples', color='purple')
    P.plot(xs, ys, label='Interpolated curve')
    P.plot(xs, xs.sin(), ' ', label='True Curve')
    P.plot(xs, Ys, label='Spline Integral')
    P.plot(xs, 1-xs.cos(), ' ', label='True Integral')
    P.legend()
    P.show()