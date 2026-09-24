"""
Project 12 - Arbitrary style transfer in real time with AdaIN (Huang & Belongie, 2017) - PyTorch
===============================================================================================

PROBLEM with project 11 (Gatys): every new image needs hundreds of optimisation
steps (seconds per image). Fast feed-forward methods fixed that but needed one
network PER STYLE.

AdaIN - Adaptive Instance Normalization - does ANY style in ONE forward pass:
    1. Encode content c and style s with a frozen VGG-19 (up to relu4_1).
    2. AdaIN(x, y) = sigma(y) * (x - mu(x)) / sigma(x) + mu(y)
       i.e. give the content features the per-channel MEAN and STD of the style
       features. (Channel statistics carry "style"; spatial layout carries content.)
    3. A learned decoder maps the re-normalised features back to an image.
    Only the decoder is trained, with
        content loss = || VGG(output) - AdaIN target ||          (at relu4_1)
        style loss   = sum over relu1_1..relu4_1 of || mu, sigma of VGG(output) - of VGG(style) ||
    At test time, alpha in [0, 1] blends content and stylised features -> a
    user-controllable style strength, and any unseen painting works.

DATA: content = natural photos (Imagenette), style = paintings (WikiArt, streamed subset).
EVALUATION: held-out content/style pairs (styles never seen in training), the
content-vs-style trade-off (alpha), and speed vs. the optimisation method of project 11.
"""
import io
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common.utils import results_dir

PROJECT = "12_adain_style_transfer"
QUICK = os.environ.get("QUICK") == "1"
DEVICE = "cuda"
CROP = 256
BATCH = 8
N_CONTENT = 400 if QUICK else 9000
N_STYLE = 400 if QUICK else 6000
TIME_BUDGET_MIN = 2 if QUICK else float(os.environ.get("TIME_BUDGET_MIN", "35"))
STYLE_WEIGHT = 10.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Pre-1900 movements only for the held-out styles shown in the published figures (public-domain artworks).
PUBLIC_DOMAIN_MOVEMENTS = {"Baroque", "Rococo", "Impressionism", "Early_Renaissance", "High_Renaissance",
                           "Northern_Renaissance", "Mannerism_Late_Renaissance", "Romanticism", "Realism", "Ukiyo_e"}


def load_images(name, config, split, n, key, public_domain_only=False, skip=0):
    """Stream n images from a Hugging Face dataset and keep them as small PIL images in memory."""
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split, streaming=True) if config else load_dataset(name, split=split, streaming=True)
    out, seen = [], 0
    names = ds.features["style"].names if public_domain_only else None
    for ex in ds.shuffle(seed=42, buffer_size=2000):
        seen += 1
        if seen <= skip:
            continue
        if public_domain_only and names[ex["style"]] not in PUBLIC_DOMAIN_MOVEMENTS:
            continue
        img = ex[key]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"]))
        img = img.convert("RGB")
        s = 320 / min(img.size)
        out.append(img.resize((max(CROP, int(img.width * s)), max(CROP, int(img.height * s))), Image.BILINEAR))
        if len(out) >= n:
            break
    return out


def split_wikiart(n_train, n_test):
    """One pass over the shuffled WikiArt stream: the first n_test public-domain paintings become the held-out
    test styles (never trained on, safe to publish); everything else goes to training."""
    from datasets import load_dataset

    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    names = ds.features["style"].names
    train, test = [], []
    for ex in ds.shuffle(seed=42, buffer_size=2000):
        img = ex["image"].convert("RGB")
        s = 320 / min(img.size)
        img = img.resize((max(CROP, int(img.width * s)), max(CROP, int(img.height * s))), Image.BILINEAR)
        if len(test) < n_test and names[ex["style"]] in PUBLIC_DOMAIN_MOVEMENTS:
            test.append(img)
        elif len(train) < n_train:
            train.append(img)
        if len(train) >= n_train and len(test) >= n_test:
            break
    return train, test


def random_crop(img):
    x = random.randint(0, img.width - CROP)
    y = random.randint(0, img.height - CROP)
    arr = np.asarray(img.crop((x, y, x + CROP, y + CROP)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def batch_of(images):
    return torch.stack([random_crop(random.choice(images)) for _ in range(BATCH)]).to(DEVICE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """Frozen VGG-19 split at relu1_1, relu2_1, relu3_1, relu4_1."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        f = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList([f[:2], f[2:7], f[7:12], f[12:21]])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, all_levels=False):
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats if all_levels else feats[-1]


def decoder():
    """Mirror of VGG up to relu4_1; nearest upsampling + reflection padding avoid checkerboard artefacts."""
    def conv(i, o, relu=True):
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(i, o, 3)]
        return layers + ([nn.ReLU(inplace=True)] if relu else [])
    return nn.Sequential(
        *conv(512, 256), nn.Upsample(scale_factor=2, mode="nearest"),
        *conv(256, 256), *conv(256, 256), *conv(256, 256), *conv(256, 128), nn.Upsample(scale_factor=2, mode="nearest"),
        *conv(128, 128), *conv(128, 64), nn.Upsample(scale_factor=2, mode="nearest"),
        *conv(64, 64), *conv(64, 3, relu=False))


def mean_std(f, eps=1e-5):
    b, c = f.shape[:2]
    v = f.view(b, c, -1)
    return v.mean(-1).view(b, c, 1, 1), (v.var(-1) + eps).sqrt().view(b, c, 1, 1)


def adain(content_f, style_f):
    cm, cs = mean_std(content_f)
    sm, ss = mean_std(style_f)
    return ss * (content_f - cm) / cs + sm


@torch.no_grad()
def stylize(enc, dec, content, style, alpha=1.0):
    fc, fs = enc(content), enc(style)
    t = alpha * adain(fc, fs) + (1 - alpha) * fc
    return dec(t).clamp(0, 1)


def losses(enc, dec, content, style):
    fc, fs_levels = enc(content), enc(style, all_levels=True)
    t = adain(fc, fs_levels[-1])
    out = dec(t)
    out_levels = enc(out, all_levels=True)
    c_loss = F.mse_loss(out_levels[-1], t)
    s_loss = 0
    for o, s in zip(out_levels, fs_levels):
        om, os_ = mean_std(o)
        sm, ss = mean_std(s)
        s_loss = s_loss + F.mse_loss(om, sm) + F.mse_loss(os_, ss)
    return c_loss, s_loss


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random.seed(0)
    torch.manual_seed(0)
    out_dir = results_dir(PROJECT)
    t0 = time.time()
    # Training photos: Flickr30k (streamed Parquet). Held-out photos shown in the published figures:
    # scikit-image's public-domain / CC0 sample images, so no licensed photo is redistributed.
    contents = load_images("lmms-lab/flickr30k", None, "test", N_CONTENT, "image")
    styles, test_styles = split_wikiart(N_STYLE, 16)
    from skimage import data as skdata
    test_contents = [Image.fromarray(f()).convert("RGB") for f in
                     (skdata.astronaut, skdata.coffee, skdata.chelsea, skdata.rocket, skdata.immunohistochemistry)]
    test_contents = [im.resize((max(CROP, int(im.width * 320 / min(im.size))), max(CROP, int(im.height * 320 / min(im.size)))))
                     for im in test_contents]
    print(f"data: {len(contents)} content, {len(styles)} style (+16 held-out) in {(time.time() - t0) / 60:.1f} min")

    enc, dec = Encoder().to(DEVICE).eval(), decoder().to(DEVICE)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 1 / (1 + 5e-5 * it))
    history, it, start = [], 0, time.time()
    while time.time() - start < TIME_BUDGET_MIN * 60:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c_loss, s_loss = losses(enc, dec, batch_of(contents), batch_of(styles))
            loss = c_loss + STYLE_WEIGHT * s_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        it += 1
        if it % 100 == 0:
            history.append({"iter": it, "content_loss": c_loss.item(), "style_loss": s_loss.item()})
            print(history[-1], flush=True)
    train_min = (time.time() - start) / 60

    # ---- held-out evaluation: unseen styles x unseen photos ----
    dec.eval()
    ev_c, ev_s = [], []
    with torch.no_grad():
        for i in range(len(test_styles)):
            c = torch.stack([random_crop(test_contents[i % len(test_contents)])]).to(DEVICE)
            s = torch.stack([random_crop(test_styles[i])]).to(DEVICE)
            cl, sl = losses(enc, dec, c, s)
            ev_c.append(cl.item()); ev_s.append(sl.item())

    # ---- speed: one 512x512 image ----
    c512 = torch.rand(1, 3, 512, 512, device=DEVICE)
    for _ in range(3):
        stylize(enc, dec, c512, c512)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20):
        stylize(enc, dec, c512, c512)
    torch.cuda.synchronize()
    ms = (time.time() - t) / 20 * 1000

    # ---- figures ----
    to_img = lambda x: x.permute(1, 2, 0).float().cpu().numpy()
    n = 5
    fig, axes = plt.subplots(n + 1, n + 1, figsize=(2.4 * (n + 1), 2.4 * (n + 1)))
    cs = torch.stack([random_crop(x) for x in test_contents[:n]]).to(DEVICE)
    ss = torch.stack([random_crop(x) for x in test_styles[:n]]).to(DEVICE)
    for ax in axes.flat:
        ax.axis("off")
    for j in range(n):
        axes[0, j + 1].imshow(to_img(ss[j])); axes[0, j + 1].set_title("unseen style", fontsize=8)
    for i in range(n):
        axes[i + 1, 0].imshow(to_img(cs[i])); axes[i + 1, 0].set_title("content", fontsize=8)
        ys = stylize(enc, dec, cs[i:i + 1].repeat(n, 1, 1, 1), ss)
        for j in range(n):
            axes[i + 1, j + 1].imshow(to_img(ys[j]))
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "adain_grid.jpg"), dpi=70); plt.close(fig)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(alphas), figsize=(2.6 * len(alphas), 2.8))
    for ax, a in zip(axes, alphas):
        ax.imshow(to_img(stylize(enc, dec, cs[:1], ss[:1], alpha=a)[0])); ax.set_title(f"alpha={a}", fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "alpha_tradeoff.jpg"), dpi=80); plt.close(fig)

    metrics = {"train_iterations": it, "train_minutes": train_min, "content_images": len(contents),
               "style_images": len(styles), "heldout_content_loss": float(np.mean(ev_c)),
               "heldout_style_loss": float(np.mean(ev_s)), "ms_per_512px_image": ms,
               "decoder_params": sum(p.numel() for p in dec.parameters()), "gpu": torch.cuda.get_device_name(0)}
    json.dump({"metrics": metrics, "history": history}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    torch.save(dec.state_dict(), os.path.expanduser("~/adain_decoder.pt"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
