"""
This script generates the average grayscale of each character.
"""

import os
import freetype
import json

# Place your font here
font_file = "NotoSansSC-Regular.ttf"
out_file = "grayscale.json"

candidate_range = [
    # (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    # (0x2000, 0x206F),  # General Punctuation
    # (0x2E80, 0x2EFF),  # CJK Radicals Supplement
    # (0x2FF0, 0x2FFF),  # Ideographic Description Characters
]
render_size = 32

result = {}

# Sanity check
if not os.path.exists(font_file):
    raise FileNotFoundError(f"Font file {font_file} not found.")

font = freetype.Face(font_file)
font.set_char_size(render_size, render_size)

for range_start, range_end in candidate_range:
    for char_code in range(range_start, range_end + 1):
        if char_code % 100 == 0:
            print(f"Processing {char_code} ({chr(char_code)})...")

        char = chr(char_code)
        font.load_char(char)
        bitmap = font.glyph.bitmap
        buf = bitmap.buffer
        width = bitmap.width
        height = bitmap.rows

        if width == 0 or height == 0:
            print(f"Skipping character {char} due to zero width or height.")
            continue

        grayscale = 0
        for i in range(height):
            for j in range(width):
                grayscale += buf[i * width + j]
        grayscale /= width * height * 255
        result[char] = grayscale

with open(out_file, "w", encoding="utf8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
