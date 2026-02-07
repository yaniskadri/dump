import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from segment_anything import sam_model_registry, SamPredictor

# ─── Configuration ───────────────────────────────────────────────────
RENDER_DPI = 300
PDF_BASE_DPI = 72
DPI_SCALE = RENDER_DPI / PDF_BASE_DPI  # ≈ 4.1667


# ═══════════════════════════════════════════════════════════════════════
# 1. ISLAND EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def get_islands_data(polygons, gap):
    """
    Group nearby polygons into islands using a buffer/merge approach.
    
    Args:
        polygons: list of Shapely Polygon objects (PDF coordinate space).
        gap: distance threshold — polygons closer than this merge into one island.
    
    Returns:
        list of [xmin, ymin, xmax, ymax] bounding boxes in PDF coords.
    """
    if not polygons:
        return []

    # Keep ALL polygons — fix invalid ones instead of dropping them
    clean = []
    for p in polygons:
        if not isinstance(p, Polygon):
            try:
                p = Polygon(p)
            except Exception:
                continue
        if p.is_empty:
            continue
        if not p.is_valid:
            p = make_valid(p)
        # make_valid can return GeometryCollections; extract polygons from those
        if isinstance(p, Polygon) and not p.is_empty:
            clean.append(p)
        elif hasattr(p, 'geoms'):
            for g in p.geoms:
                if isinstance(g, Polygon) and not g.is_empty:
                    clean.append(g)

    if not clean:
        return []

    print(f"  [islands] {len(polygons)} input → {len(clean)} valid polygons")

    # Buffer, merge, split into groups
    buffered = [p.buffer(gap) for p in clean]
    merged = unary_union(buffered)

    if isinstance(merged, Polygon):
        island_shapes = [merged]
    else:
        island_shapes = list(merged.geoms)

    # For each island, compute bbox from the ORIGINAL member polygons (no buffer bloat)
    islands = []
    used = 0
    for island_shape in island_shapes:
        members = [p for p in clean if island_shape.intersects(p)]
        if not members:
            continue
        used += len(members)
        all_bounds = [m.bounds for m in members]  # (xmin, ymin, xmax, ymax)
        xmin = min(b[0] for b in all_bounds)
        ymin = min(b[1] for b in all_bounds)
        xmax = max(b[2] for b in all_bounds)
        ymax = max(b[3] for b in all_bounds)
        islands.append([xmin, ymin, xmax, ymax])

    print(f"  [islands] → {len(islands)} islands  "
          f"({used} polygons assigned, {len(clean) - used} orphaned)")
    return islands


# ═══════════════════════════════════════════════════════════════════════
# 2. SAM HIERARCHICAL DETECTION  (with proper DPI scaling)
# ═══════════════════════════════════════════════════════════════════════

def run_sam_hierarchical(image, polygons, gap=3, sam_checkpoint=r"..\Models\sam_vit_b_01ec64.pth"):
    """
    Run SAM on each island with properly scaled coordinates.
    
    Polygons are in PDF space (72 DPI), image is rendered at RENDER_DPI.
    Only uses box prompts — one per island.
    
    Returns:
        (final_masks, image)
    """
    # Setup SAM
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Build islands (list of bbox in PDF space)
    islands = get_islands_data(polygons, gap)
    print(f"Islands detected: {len(islands)}")

    final_masks = []

    for idx, pdf_box in enumerate(islands):
        # Scale PDF coords → pixel coords
        pdf_box = np.array(pdf_box)
        pixel_box = pdf_box * DPI_SCALE

        try:
            masks, scores, _ = predictor.predict(
                box=pixel_box,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            chosen_mask = masks[best]
        except Exception as e:
            print(f"  [!] SAM failed on island {idx}: {e}")
            continue

        final_masks.append({
            'mask': chosen_mask,
            'box_pixel': pixel_box,
            'box_pdf': pdf_box,
            'score': float(scores[best]),
            'id': idx,
        })

    return final_masks, image


# ═══════════════════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def _mask_to_tight_bbox(mask):
    """Return (xmin, ymin, xmax, ymax) of the True region, or None."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (xs.min(), ys.min(), xs.max(), ys.max())


def visualize_results(image, sam_results, islands_data=None,
                      save_path="detection_output.png", figsize=(24, 16)):
    """
    Rich visualization showing:
        • Original image
        • Island bounding boxes (dashed blue)
        • SAM mask overlays (translucent color fills)
        • Tight bounding box around each SAM mask (solid color)
        • Labels with island id and SAM confidence
    
    Args:
        image:        BGR numpy array (rendered at RENDER_DPI)
        sam_results:  list of dicts from run_sam_hierarchical()
        islands_data: (optional) list of island dicts for drawing island boxes
        save_path:    where to save the figure (None to skip saving)
        figsize:      matplotlib figure size
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left panel: island boxes on clean image ──────────────────────
    axes[0].imshow(img_rgb)
    axes[0].set_title("Islands (vector grouping)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    if islands_data:
        for i, bbox in enumerate(islands_data):
            bx = np.array(bbox) * DPI_SCALE
            w, h = bx[2] - bx[0], bx[3] - bx[1]
            rect = mpatches.Rectangle(
                (bx[0], bx[1]), w, h,
                linewidth=2, edgecolor='dodgerblue',
                facecolor='dodgerblue', alpha=0.08, linestyle='--',
            )
            axes[0].add_patch(rect)
            axes[0].text(bx[0], bx[1] - 4, f'island {i}',
                         color='white', fontsize=7, fontweight='bold',
                         bbox=dict(boxstyle='square,pad=0.15',
                                   facecolor='dodgerblue', alpha=0.7, edgecolor='none'))

    # ── Right panel: SAM detections with masks ───────────────────────
    overlay = img_rgb.astype(np.float64).copy()

    for res in sam_results:
        np.random.seed(res['id'] * 37 + 7)
        color = np.random.randint(60, 255, (3,)).astype(np.float64)
        mask = res['mask']
        overlay[mask] = overlay[mask] * 0.55 + color * 0.45

    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title("SAM Detections (masks + boxes)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    for res in sam_results:
        np.random.seed(res['id'] * 37 + 7)
        color_01 = np.random.randint(60, 255, (3,)) / 255.0

        # Tight bbox from mask (most accurate indicator of what SAM actually found)
        tight = _mask_to_tight_bbox(res['mask'])
        if tight:
            tx0, ty0, tx1, ty1 = tight
            rect = mpatches.Rectangle(
                (tx0, ty0), tx1 - tx0, ty1 - ty0,
                linewidth=2, edgecolor=color_01, facecolor='none', linestyle='-',
            )
            axes[1].add_patch(rect)

        # Island prompt box (dashed, thinner) so you can compare
        bx = res['box_pixel']
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        rect2 = mpatches.Rectangle(
            (bx[0], bx[1]), w, h,
            linewidth=1.2, edgecolor=color_01, facecolor='none', linestyle='--',
        )
        axes[1].add_patch(rect2)

        # Label
        label_y = (tight[1] if tight else bx[1]) - 4
        label_x = tight[0] if tight else bx[0]
        score_str = f"{res['score']:.2f}" if 'score' in res else ""
        axes[1].text(label_x, label_y,
                     f"id{res['id']} {score_str}",
                     color='white', fontsize=7, fontweight='bold',
                     bbox=dict(boxstyle='square,pad=0.15',
                               facecolor=color_01, alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()

    # ── Console summary ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" SAM DETECTION SUMMARY  ({len(sam_results)} objects)")
    print(f"{'='*65}")
    for res in sam_results:
        bx = res['box_pixel']
        tight = _mask_to_tight_bbox(res['mask'])
        t_str = (f"tight=({tight[0]:.0f},{tight[1]:.0f})-({tight[2]:.0f},{tight[3]:.0f})"
                 if tight else "no mask pixels")
        score = res.get('score', 0)
        print(f"  [{res['id']:3d}]  prompt_box=({bx[0]:.0f},{bx[1]:.0f})-"
              f"({bx[2]:.0f},{bx[3]:.0f})  {t_str}  score={score:.3f}")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════
# 4. MAIN SCRIPT
# ═══════════════════════════════════════════════════════════════════════

from Crop_Pipeline.vector_utils import extract_segments_from_page, extract_text_blocks
from Crop_Pipeline.vector_engine import extract_shapes
import fitz

# File configuration
INPUT_PDF_PATH = r"C:\Users\ykadr\Pictures\DB-Test\wd_indiv\wd_0009.pdf"

doc = fitz.open(INPUT_PDF_PATH)
page = doc[0]

# Step 1: Extract vector data (in PDF coordinate space, 72 DPI)
segments = extract_segments_from_page(page)
text_blocks = extract_text_blocks(page)
polygons = extract_shapes(segments)
print(f"Extracted {len(polygons)} polygons from PDF")

# Step 2: Build islands (still in PDF space)
GAP = 3
islands = get_islands_data(polygons, GAP)
print(f"Grouped into {len(islands)} islands (gap={GAP})")

# Step 3: Rasterize at high DPI for SAM
print(f"Rendering page at {RENDER_DPI} DPI  (scale factor = {DPI_SCALE:.2f}x) ...")
pix = page.get_pixmap(dpi=RENDER_DPI)
image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
print(f"Image size: {image.shape[1]}×{image.shape[0]} px")

# Step 4: Run SAM with properly scaled coordinates
print("Running SAM...")
masks, _ = run_sam_hierarchical(image, polygons, gap=GAP)
print(f"SAM returned {len(masks)} detections")

# Step 5: Visualize everything
visualize_results(image, masks, islands_data=islands, save_path="detection_output.png")

