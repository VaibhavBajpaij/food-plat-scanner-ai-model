"""
Create a synthetic, legally unencumbered (100% public domain / procedurally generated)
test food image with calibrated plate geometry and reference fiducials.
"""

import os
import math
from PIL import Image, ImageDraw

def generate_calibrated_plate_sample(output_path: str):
    width, height = 512, 512
    img = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # 1. Background table surface (subtle wood grain texture)
    for y in range(0, height, 4):
        shade = 225 + int(10 * math.sin(y / 15.0))
        draw.line([(0, y), (width, y)], fill=(shade, shade - 15, shade - 30))

    # 2. Plate rim (White ceramic dinner plate, 24.0 cm nominal diameter)
    # Circle centered at (256, 256), radius 190 px (so diameter = 380 px -> scale = 380px / 24cm = 15.83 px/cm)
    plate_bbox = [256 - 190, 256 - 190, 256 + 190, 256 + 190]
    draw.ellipse(plate_bbox, fill=(250, 250, 252), outline=(210, 210, 215), width=4)
    # Inner plate well
    draw.ellipse([256 - 145, 256 - 145, 256 + 145, 256 + 145], fill=(255, 255, 255), outline=(230, 230, 235), width=2)

    # 3. Reference scale marker (Standard coin / fiducial, 2.4 cm diameter, at (130, 256))
    # 2.4 cm * 15.83 px/cm = 38 px diameter -> radius 19 px
    coin_bbox = [130 - 19, 256 - 19, 130 + 19, 256 + 19]
    draw.ellipse(coin_bbox, fill=(205, 170, 75), outline=(160, 130, 45), width=2)
    draw.text((120, 250), "REF", fill=(80, 60, 10))

    # 4. Food Item: Red Apple with volumetric shading (centered at (280, 256))
    # Apple radius ~60 px (diameter 120 px -> ~7.6 cm diameter, standard medium apple)
    # Draw shadow
    draw.ellipse([280 - 65, 256 + 30, 280 + 65, 256 + 75], fill=(210, 210, 215))

    # Draw shaded body (spherical gradient simulation)
    center_x, center_y = 280, 256
    radius = 58
    for r in range(radius, 0, -2):
        factor = r / radius
        r_col = int(210 + 40 * (1 - factor))
        g_col = int(25 * factor)
        b_col = int(25 * factor)
        draw.ellipse([center_x - r + int(12 * (1 - factor)),
                      center_y - r - int(8 * (1 - factor)),
                      center_x + r + int(12 * (1 - factor)),
                      center_y + r - int(8 * (1 - factor))],
                     fill=(r_col, g_col, b_col))

    # Highlight
    draw.ellipse([center_x - 20, center_y - 30, center_x - 5, center_y - 15], fill=(255, 140, 140))

    # Apple stem and small leaf
    draw.line([(center_x, center_y - radius + 5), (center_x - 5, center_y - radius - 15)], fill=(90, 55, 25), width=4)
    draw.polygon([(center_x - 5, center_y - radius - 10),
                  (center_x + 18, center_y - radius - 22),
                  (center_x + 8, center_y - radius - 8)], fill=(40, 150, 40))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return {
        "output_path": output_path,
        "width": width,
        "height": height,
        "nominal_plate_diameter_cm": 24.0,
        "measured_plate_diameter_px": 380,
        "ground_truth_food": "apple_fresh_raw",
        "ground_truth_diameter_cm": 7.6,
        "ground_truth_volume_cm3": round((4.0/3.0) * math.pi * ((7.6/2.0)**3), 2),
        "license": "Public Domain (Procedurally Generated)"
    }

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw/sample_plate_apple.png"
    meta = generate_calibrated_plate_sample(out)
    print("Generated sample:", meta)
