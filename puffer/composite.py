"""Compose transparent renders onto a studio gradient.
python composite.py <outdir> <sheet.png> <name1> [name2 ...]"""
import sys, os
from PIL import Image

outdir, sheet, names = sys.argv[1], sys.argv[2], sys.argv[3:]


def gradient(w, h, top=(246, 247, 249), bottom=(214, 219, 226)):
    g = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(h - 1, 1)
        g.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    return g.resize((w, h))


ims = [Image.open(os.path.join(outdir, n + ".png")).convert("RGBA") for n in names]
w, h = ims[0].size
out = gradient(w * len(ims), h)
for i, im in enumerate(ims):
    out.paste(im, (i * w, 0), im)
out.save(sheet)
print("saved", sheet, out.size)
