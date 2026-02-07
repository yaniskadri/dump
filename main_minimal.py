"""
SIMPLE MAIN SCRIPT - Boxes only, no masks
"""

from Crop_Pipeline.vector_utils import extract_segments_from_page
from Crop_Pipeline.vector_engine import extract_shapes, get_islands_data
from Crop_Pipeline.sam_v2 import run_sam_hierarchical

from minimal_viz import visualize_sam_boxes_only, check_why_only_one_island

import fitz
import numpy as np

INPUT_PDF_PATH = r"C:\Users\ykadr\Pictures\DB-Test\wd_indiv\wd_0009.pdf"

# Load and extract
doc = fitz.open(INPUT_PDF_PATH)
page = doc[0]

segments = extract_segments_from_page(page)
polygons = extract_shapes(segments)
print(f"Extracted {len(polygons)} polygons")

# Create islands
GAP = 3
islands_data = get_islands_data(polygons, GAP)
print(f"Created {len(islands_data)} islands with gap={GAP}")

# Render
pix = page.get_pixmap(dpi=300)  
image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)

# Run SAM
print("Running SAM...")
masks, _ = run_sam_hierarchical(image, polygons)
print(f"SAM returned {len(masks)} results")

# 🔍 DEBUG: Check why only one
check_why_only_one_island(islands_data, masks)

# Visualize - BOXES ONLY
visualize_sam_boxes_only(image, masks, save_path="boxes_only.png")

print("\n✓ Done! Check boxes_only.png")
