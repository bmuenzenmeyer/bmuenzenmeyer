"""
Generate the profile README's brand images into ../assets.

Two visual languages, one profile:
  * brianmuenzenmeyer.com  -> black, 1px/6px dot grid, Bayer 8x8 ordered
                              dither (3 grey levels), classic Tetris pieces
  * approachableopensource.com -> the hero css-doodle: 52x19 grid, rainbow band
                              hsl(x*6, 80%, 60%), random grey corner ticks,
                              random opacity, Urbanist + Source Serif 4

Usage:  python3 scripts/generate-images.py [--seed N]
Requires Pillow and numpy.
"""

import argparse
import colorsys
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
FONTS = Path(__file__).resolve().parent / "fonts"

W = 1600  # rendered at ~830px on GitHub, so ~2x for crisp retina display

# --- brianmuenzenmeyer.com -------------------------------------------------

TETRIS = {  # cells (col,row) + colors straight from /js/tetris-component.js
    "i": ([(0, 0), (0, 1), (0, 2), (0, 3)], "#00f0f0"),
    "o": ([(0, 0), (1, 0), (0, 1), (1, 1)], "#f0f000"),
    "t": ([(0, 0), (1, 0), (2, 0), (1, 1)], "#a000f0"),
    "s": ([(1, 0), (2, 0), (0, 1), (1, 1)], "#00f000"),
    "z": ([(0, 0), (1, 0), (1, 1), (2, 1)], "#f00000"),
    "j": ([(1, 0), (1, 1), (0, 2), (1, 2)], "#0000f0"),
    "l": ([(0, 0), (0, 1), (0, 2), (1, 2)], "#f0a000"),
}
BLOCK = 40

BAYER8 = np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ]
) / 64.0


def bayer_dither(gray, levels=3, cell=2):
    """Ordered-dither a float [0,1] array to `levels` grey levels.
    `cell` makes each dither pixel cell x cell so it survives GitHub's downscale."""
    h, w = gray.shape
    small = gray[::cell, ::cell]
    sh, sw = small.shape
    tile = np.tile(BAYER8, (sh // 8 + 1, sw // 8 + 1))[:sh, :sw]
    steps = levels - 1
    q = np.floor(small * steps + tile).clip(0, steps) / steps
    out = np.kron(q, np.ones((cell, cell)))[:h, :w]
    return (out * 255).astype(np.uint8)


def dot_grid(img, space=12, size=2, color=(255, 255, 255), alpha=70):
    """The site's radial-gradient dot background (1px dots every 6px, at 2x)."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for y in range(space // 2, img.height, space):
        for x in range(space // 2, img.width, space):
            d.rectangle([x, y, x + size - 1, y + size - 1], fill=color + (alpha,))
    img.alpha_composite(layer)


def draw_piece(d, name, x, y, block=BLOCK, rot=0):
    cells, color = TETRIS[name]
    for _ in range(rot % 4):  # rotate 90deg clockwise
        maxr = max(r for _, r in cells)
        cells = [(maxr - r, c) for c, r in cells]
    for c, r in cells:
        x0, y0 = x + c * block, y + r * block
        d.rectangle([x0, y0, x0 + block - 1, y0 + block - 1], fill=color, outline="#000000", width=3)


def piece_size(name, rot=0, block=BLOCK):
    cells, _ = TETRIS[name]
    cols = max(c for c, _ in cells) + 1
    rows = max(r for _, r in cells) + 1
    if rot % 2:
        cols, rows = rows, cols
    return cols * block, rows * block


def scatter_pieces(d, rng, box, count, avoid=(), block=BLOCK):
    """Drop pieces at random, non-overlapping, some peeking past the top edge."""
    placed = list(avoid)
    x0, y0, x1, y1 = box
    names = list(TETRIS)
    rng.shuffle(names)
    tries = 0
    while count and tries < 400:
        tries += 1
        n = names[count % len(names)]
        rot = rng.randrange(4)
        pw, ph = piece_size(n, rot, block)
        x = rng.randrange(x0, max(x0 + 1, x1 - pw))
        y = rng.randrange(y0, max(y0 + 1, y1 - ph))
        rect = (x - 16, y - 16, x + pw + 16, y + ph + 16)
        if any(not (rect[2] < a[0] or rect[0] > a[2] or rect[3] < a[1] or rect[1] > a[3]) for a in placed):
            continue
        draw_piece(d, n, x, y, block, rot)
        placed.append(rect)
        count -= 1


def noise_field(h, w, rng, octaves=((80, 0.55), (28, 0.3), (9, 0.15))):
    """Cheap smooth value noise for dither texture."""
    acc = np.zeros((h, w))
    for scale, amp in octaves:
        gh, gw = h // scale + 2, w // scale + 2
        g = np.array(rng.random((gh, gw)))
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        acc += np.asarray(img, dtype=float) / 255 * amp
    return acc


def inter(size, weight="Bold"):
    for p in (FONTS / f"Inter-{weight}.otf", Path(f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf")):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    raise FileNotFoundError("Inter font missing")


def hero(rng, nrng):
    h = 560
    yy, xx = np.mgrid[0:h, 0:W]
    # a stage spotlight falling from upper right, textured with noise
    cx, cy = W * 0.86, -h * 0.1
    dist = np.sqrt(((xx - cx) / (W * 0.5)) ** 2 + ((yy - cy) / (h * 1.3)) ** 2)
    light = np.clip(1.05 - dist, 0, 1) ** 1.5
    tex = noise_field(h, W, nrng)
    gray = np.clip(light * (0.55 + 0.6 * tex), 0, 1)
    # keep the type sitting in shadow so it stays legible
    shade = np.clip(1 - np.maximum((xx - 1080) / 260, 0), 0, 1) * np.clip(1 - np.maximum((yy - 330) / 80, 0), 0, 1)
    gray *= 1 - 0.85 * shade
    img = Image.fromarray(bayer_dither(gray)).convert("RGBA")
    dot_grid(img, alpha=55)
    d = ImageDraw.Draw(img)

    title = inter(104)
    d.text((64, 70), "Brian Muenzenmeyer", font=title, fill="white")
    sub = inter(40, "SemiBold")
    d.text((68, 206), "Community builder. Engineer. Bald guy.", font=sub, fill="white")
    small = inter(30, "Medium")
    d.text((68, 268), "Node.js website maintainer  ·  author of Approachable Open Source", font=small, fill=(200, 200, 200))

    avoid = [(40, 60, 1170, 200), (40, 200, 900, 250), (40, 250, 1050, 320)]
    scatter_pieces(d, rng, (-30, -90, W + 10, h + 20), 14, avoid=avoid)
    return img.convert("RGB")


def section(label, rng, nrng, slug):
    h = 200
    yy, xx = np.mgrid[0:h, 0:W]
    ramp = (xx / W) ** 1.4 * 0.75
    tex = noise_field(h, W, nrng, octaves=((40, 0.6), (12, 0.4)))
    gray = np.clip(ramp * (0.6 + 0.6 * tex), 0, 1)
    img = Image.fromarray(bayer_dither(gray)).convert("RGBA")
    dot_grid(img, alpha=45)
    d = ImageDraw.Draw(img)
    f = inter(72)
    tw = d.textlength(label, font=f)
    pad = 26
    # black plate behind the label keeps it legible over the dither
    d.rectangle([48, 52, 48 + tw + pad * 2, 148], fill="black")
    d.text((48 + pad, 58), label, font=f, fill="white")
    scatter_pieces(d, rng, (int(48 + tw + pad * 2 + 80), -50, W - 40, h - 20), 3, block=36)
    img.convert("RGB").save(OUT / f"section-{slug}.png", optimize=True)


def footer(rng):
    """The pieces finally land: a Tetris floor on the dot grid."""
    h = 240
    img = Image.new("RGBA", (W, h), (0, 0, 0, 255))
    dot_grid(img, alpha=70)
    d = ImageDraw.Draw(img)
    b = BLOCK
    cols, rows = W // b, 3
    floor = np.zeros((rows, cols), dtype=object)
    # fill the bottom rows with piece-colored blocks, leaving a few holes
    names = list(TETRIS)
    for r in range(rows):
        for c in range(cols):
            if rng.random() < (0.94 if r > 0 else 0.55):
                floor[r, c] = TETRIS[names[(c // 3 + r * 2 + rng.randrange(2)) % 7]][1]
    oy = h - rows * b
    for r in range(rows):
        for c in range(cols):
            if floor[r, c]:
                x0, y0 = c * b, oy + r * b
                d.rectangle([x0, y0, x0 + b - 1, y0 + b - 1], fill=floor[r, c], outline="#000", width=3)
    # one last piece still falling
    draw_piece(d, "t", int(W * 0.62), 20, b, 2)
    draw_piece(d, "i", int(W * 0.18), -70, b, 0)
    return img.convert("RGB")


# --- approachableopensource.com --------------------------------------------

OPACITIES = [1, 0.6, 0.3, 0.1, 0, 0, 0.2, 0.4, 0.8]


def hsl(hdeg, s, l):
    r, g, b = colorsys.hls_to_rgb((hdeg % 360) / 360, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def aos_grid(img, rng, cols=52, rows=19, cell=22, gap=7, ox=None, oy=None, tick=(128, 128, 128), band=(8, 12)):
    """Port of the hero's css-doodle rule onto a PIL canvas."""
    step = cell + gap
    ox = (img.width - cols * step + gap) // 2 if ox is None else ox
    oy = (img.height - rows * step + gap) // 2 if oy is None else oy
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            a = rng.choice(OPACITIES)
            if a == 0:
                continue
            px, py = ox + (x - 1) * step, oy + (y - 1) * step
            if band[0] <= y <= band[1]:
                d.rectangle([px, py, px + cell - 1, py + cell - 1], fill=hsl(x * 6, 0.8, 0.6) + (int(255 * a),))
                continue
            top, left = rng.random() < 0.5, rng.random() < 0.5
            if (y < 4 or y > 16) and rng.random() < 0.5:
                top = left = False
            col = tick + (int(255 * a),)
            if top:
                d.rectangle([px, py, px + cell - 1, py + 1], fill=col)
            if left:
                d.rectangle([px, py, px + 1, py + cell - 1], fill=col)
    img.alpha_composite(layer)


def aos_banner(rng_seed, theme):
    rng = random.Random(rng_seed)
    h = 640
    dark = theme == "dark"
    bg = (9, 11, 17) if dark else (255, 255, 255)
    img = Image.new("RGBA", (W, h), bg + (255,))
    aos_grid(img, rng, tick=(128, 128, 128) if dark else (160, 166, 184))
    if not dark:  # mute the band slightly behind dark type
        pass
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(str(FONTS / "urbanist-latin-500-normal.ttf"), 168)
    serif = ImageFont.truetype(str(FONTS / "source-serif-4-latin-400-normal.ttf"), 40)
    fg = (255, 255, 255) if dark else (9, 11, 17)
    sub = (195, 202, 219) if dark else (61, 70, 99)
    d.text((70, 150), "APPROACHABLE", font=title, fill=fg)
    d.text((70, 320), "OPEN SOURCE", font=title, fill=fg)
    d.text((76, 520), "By Brian Muenzenmeyer. A brief book with a big vision.", font=serif, fill=sub)
    return img.convert("RGB")


def aos_strip(seed):
    """Transparent rainbow strip, works on GitHub light and dark."""
    rng = random.Random(seed)
    cols, cell, gap = 52, 22, 7
    step = cell + gap
    img = Image.new("RGBA", (cols * step - gap, 3 * step - gap), (0, 0, 0, 0))
    aos_grid(img, rng, cols=cols, rows=3, cell=cell, gap=gap, ox=0, oy=0, band=(1, 3))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    seed = ap.parse_args().seed
    OUT.mkdir(exist_ok=True)

    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    hero(rng, nrng).save(OUT / "hero.png", optimize=True)
    for i, (label, slug) in enumerate(
        [("Keeping me engaged lately", "lately"), ("Bio", "bio"), ("From now and then", "now-and-then")]
    ):
        section(label, random.Random(seed + i + 1), np.random.default_rng(seed + i + 1), slug)
    footer(random.Random(seed + 9)).save(OUT / "footer.png", optimize=True)
    for theme in ("dark", "light"):
        aos_banner(seed, theme).save(OUT / f"aos-banner-{theme}.png", optimize=True)
    aos_strip(seed + 3).save(OUT / "aos-strip.png", optimize=True)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
