"""Compare gsplat forward and backward against fixed inputs, bit for bit.

Training comparisons confound kernel numerics with densification decisions and
view sampling. This removes all of that: identical tensors built from a fixed
seed on CPU, moved to the device, one forward and one backward, results dumped
raw. Any difference between machines is the kernels.
"""
import sys, numpy as np, torch
from gsplat import rasterization

out = sys.argv[1]
g = torch.Generator().manual_seed(1234)          # CPU generator = identical on both
N = 20000
means  = torch.randn(N, 3, generator=g) * 0.8
quats  = torch.nn.functional.normalize(torch.randn(N, 4, generator=g), dim=-1)
scales = torch.rand(N, 3, generator=g) * 0.06 + 0.01
opac   = torch.rand(N, generator=g)
colors = torch.rand(N, 3, generator=g)
viewmat = torch.eye(4)[None]; viewmat[0, 2, 3] = 3.5
K = torch.tensor([[[420., 0., 256.], [0., 420., 192.], [0., 0., 1.]]])

d = torch.device("cuda")
means, quats, scales = means.to(d), quats.to(d), scales.to(d)
opac, colors = opac.to(d), colors.to(d)
viewmat, K = viewmat.to(d), K.to(d)
for t in (means, scales, opac, colors):
    t.requires_grad_(True)

img, alpha, meta = rasterization(means, quats, scales, opac, colors,
                                 viewmat, K, 512, 384)
loss = (img ** 2).mean()
loss.backward()

np.savez(out,
         img=img.detach().cpu().numpy().astype(np.float64),
         alpha=alpha.detach().cpu().numpy().astype(np.float64),
         g_means=means.grad.cpu().numpy().astype(np.float64),
         g_scales=scales.grad.cpu().numpy().astype(np.float64),
         g_opac=opac.grad.cpu().numpy().astype(np.float64),
         g_colors=colors.grad.cpu().numpy().astype(np.float64),
         loss=np.float64(loss.item()))
print(f"{torch.cuda.get_device_name(0)}  loss={loss.item():.10f}  "
      f"img_mean={img.mean().item():.10f}")
