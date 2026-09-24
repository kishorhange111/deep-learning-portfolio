"""
Project 11 - Neural Style Transfer (Gatys et al., 2015) in PyTorch
==================================================================

TASK: repaint a photo in the style of a painting ("the astronaut, painted by Van Gogh").

KEY IDEA - a pretrained CNN separates *content* from *style*:
    * Content  = the feature maps of a deep VGG-19 layer (conv4_2): WHAT is where.
    * Style    = the Gram matrices (channel-by-channel correlations) of several
                 layers (conv1_1 ... conv5_1): WHICH textures / colours / strokes
                 co-occur, regardless of position.
    We start from the photo and *optimise the pixels themselves* (not a network!)
    to minimise:  alpha * content_loss + beta * style_loss + gamma * total_variation.

IMPROVEMENTS over the Keras example:
    * L-BFGS optimiser (quasi-Newton; converges in ~300 steps instead of thousands of SGD steps)
    * Total-variation loss to suppress high-frequency noise
    * Several content/style pairs + a content-vs-style weight sweep, with timings
      (sets up the comparison with project 12, AdaIN, which does this in one forward pass)

Images: content photos from scikit-image's sample data (public domain / CC0);
style paintings are public-domain works from Wikimedia Commons.
"""
import io
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common.utils import results_dir

PROJECT = "11_neural_style_transfer"
QUICK = os.environ.get("QUICK") == "1"
SIZE = 256 if QUICK else 512
STEPS = 20 if QUICK else 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIKI = "https://commons.wikimedia.org/wiki/Special:FilePath/{}?width=800"
STYLES = {
    "starry_night": "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "great_wave": "Tsunami_by_hokusai_19th_century.jpg",
    "impression_sunrise": "Monet_-_Impression,_Sunrise.jpg",
    "the_scream": "Edvard_Munch,_1893,_The_Scream,_oil,_tempera_and_pastel_on_cardboard,_91_x_73_cm,_National_Gallery_of_Norway.jpg",
}
CONTENT_LAYER = "conv4_2"
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def fetch_style(name):
    req = urllib.request.Request(WIKI.format(STYLES[name]), headers={"User-Agent": "portfolio-research/1.0"})
    return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=60).read())).convert("RGB")


def content_images():
    from skimage import data

    return {"astronaut": Image.fromarray(data.astronaut()), "coffee": Image.fromarray(data.coffee()),
            "cat": Image.fromarray(data.chelsea())}


def to_tensor(img, size):
    img = img.resize((size, int(size * img.height / img.width)), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return x.contiguous().to(DEVICE)          # L-BFGS flattens gradients with .view(), which needs contiguous memory


class VGGFeatures(torch.nn.Module):
    """VGG-19 conv layers, returning the activations we need, by name."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        self.body = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(DEVICE)
        for p in self.body.parameters():
            p.requires_grad_(False)
        names, block, conv = [], 1, 0
        for layer in self.body:
            if isinstance(layer, torch.nn.Conv2d):
                conv += 1
                names.append(f"conv{block}_{conv}")
            elif isinstance(layer, torch.nn.MaxPool2d):
                names.append(f"pool{block}")
                block, conv = block + 1, 0
            else:
                names.append(f"relu{block}_{conv}")
        self.names = names

    def forward(self, x):
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        out = {}
        for name, layer in zip(self.names, self.body):
            x = layer(x)
            out[name] = x
            if name == "conv5_1":
                break
        return out


def gram(f):
    b, c, h, w = f.shape
    f = f.view(c, h * w)
    return f @ f.t() / (c * h * w)


def stylize(vgg, content, style, style_weight=1e6, tv_weight=1e-5, steps=STEPS):
    with torch.no_grad():
        c_feats = vgg(content)
        s_grams = {l: gram(f) for l, f in vgg(style).items() if l in STYLE_LAYERS}
    x = content.clone().requires_grad_(True)        # start from the photo -> keeps structure, faster
    opt = torch.optim.LBFGS([x], max_iter=steps, line_search_fn="strong_wolfe")
    history = []

    def closure():
        opt.zero_grad()
        feats = vgg(x.clamp(0, 1))
        c_loss = F.mse_loss(feats[CONTENT_LAYER], c_feats[CONTENT_LAYER])
        s_loss = sum(F.mse_loss(gram(feats[l]), s_grams[l]) for l in STYLE_LAYERS) / len(STYLE_LAYERS)
        tv = (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        loss = c_loss + style_weight * s_loss + tv_weight * tv * 1e3
        loss.backward()
        history.append([c_loss.item(), s_loss.item()])
        return loss

    start = time.time()
    opt.step(closure)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    return x.detach().clamp(0, 1), time.time() - start, history


def to_pil(x):
    return Image.fromarray((x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    out = results_dir(PROJECT)
    vgg = VGGFeatures()
    contents = content_images()
    styles = {}
    for name in (list(STYLES)[:2] if QUICK else STYLES):
        try:
            styles[name] = fetch_style(name)
        except Exception as e:
            print(f"skipping style {name}: {e}")
    timings = {}

    # 1) grid: every content x every style
    fig, axes = plt.subplots(len(contents), len(styles) + 1, figsize=(3.2 * (len(styles) + 1), 3.2 * len(contents)))
    for i, (cname, cimg) in enumerate(contents.items()):
        c = to_tensor(cimg, SIZE)
        axes[i, 0].imshow(to_pil(c)); axes[i, 0].set_title(f"content: {cname}", fontsize=9)
        for j, (sname, simg) in enumerate(styles.items()):
            s = to_tensor(simg, SIZE)
            y, secs, hist = stylize(vgg, c, s)
            timings[f"{cname}+{sname}"] = secs
            to_pil(y).save(os.path.join(out, f"{cname}_{sname}.jpg"), quality=90)
            axes[i, j + 1].imshow(to_pil(y)); axes[i, j + 1].set_title(f"{sname} ({secs:.1f}s)", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "style_grid.jpg"), dpi=80); plt.close(fig)

    # 2) content-vs-style trade-off for one pair
    c, s = to_tensor(contents["astronaut"], SIZE), to_tensor(next(iter(styles.values())), SIZE)
    weights = [1e4, 1e5, 1e6, 1e7]
    fig, axes = plt.subplots(1, len(weights), figsize=(3.2 * len(weights), 3.4))
    for ax, w in zip(axes, weights):
        y, _, _ = stylize(vgg, c, s, style_weight=w)
        ax.imshow(to_pil(y)); ax.set_title(f"style weight {w:.0e}", fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "style_weight_sweep.jpg"), dpi=80); plt.close(fig)

    # thumbnails of the style images (public domain) for the README
    fig, axes = plt.subplots(1, len(styles), figsize=(3 * len(styles), 3))
    for ax, (n, im) in zip(np.atleast_1d(axes), styles.items()):
        ax.imshow(im); ax.set_title(n, fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "styles.jpg"), dpi=70); plt.close(fig)

    metrics = {"resolution": SIZE, "lbfgs_steps": STEPS, "pairs": len(timings),
               "mean_seconds_per_image": float(np.mean(list(timings.values()))),
               "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"}
    json.dump({"metrics": metrics, "timings": timings}, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
