"""python crop_compare.py out.png label1=img1.png [label2=img2.png ...] — crop neck area side by side."""
import sys
from PIL import Image, ImageDraw, ImageFont

out, items = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
tiles = []
for label, path in items:
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    box = (int(w * 0.2), int(h * 0.055), int(w * 0.8), int(h * 0.38))
    bg = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (236, 238, 242, 255))
    bg.alpha_composite(im.crop(box))
    ImageDraw.Draw(bg).text((12, 10), label, fill=(30, 30, 30, 255), font=ImageFont.load_default(28))
    tiles.append(bg.convert("RGB"))
W = sum(t.width for t in tiles)
sheet = Image.new("RGB", (W, tiles[0].height), "white")
x = 0
for t in tiles:
    sheet.paste(t, (x, 0)); x += t.width
sheet.save(out)
