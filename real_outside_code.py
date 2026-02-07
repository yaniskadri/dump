import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from segment_anything import sam_model_registry, SamPredictor

# ─── Configuration ───────────────────────────────────────────────────
RENDER_DPI = 300
PDF_BASE_DPI = 72
DPI_SCALE = RENDER_DPI / PDF_BASE_DPI  # ≈ 4.1667


# ═══════════════════════════════════════════════════════════════════════
# 1. ISLAND EXTRACTION  (replaces Crop_Pipeline.vector_engine.get_islands_data)
# ═══════════════════════════════════════════════════════════════════════

def get_islands_data(polygons, gap):
    """
    Group nearby polygons into islands using a buffer/merge approach.
    
    Args:
        polygons: list of Shapely Polygon objects (PDF coordinate space).
        gap: distance threshold for merging nearby polygons.
    
    Returns:
        list of dicts with keys:
            'bbox'      – [xmin, ymin, xmax, ymax] in PDF coords
            'members'   – all member polygons (sorted largest first)
            'centroid'  – (x, y) centroid of the merged island shape
    """
    if not polygons:
        return []

    # Ensure we have proper Shapely Polygons (handles both raw coords & Polygon objects)
    clean = []
    for p in polygons:
        if isinstance(p, Polygon):
            if p.is_valid and not p.is_empty:
                clean.append(p)
        else:
            try:
                sp = Polygon(p)
                if sp.is_valid and not sp.is_empty:
                    clean.append(sp)
            except Exception:
                continue

    if not clean:
        return []

    buffered = [p.buffer(gap) for p in clean]
    merged = unary_union(buffered)

    if isinstance(merged, Polygon):
        island_shapes = [merged]
    else:
        island_shapes = list(merged.geoms)

    islands_data = []
    for island_shape in island_shapes:
        members = [p for p in clean if island_shape.intersects(p)]
        members.sort(key=lambda x: x.area, reverse=True)

        # Use the island centroid (not just largest member) for the SAM positive point
        centroid = island_shape.centroid
        islands_data.append({
            'bbox': list(island_shape.buffer(-gap).bounds if not island_shape.buffer(-gap).is_empty
                         else island_shape.bounds),
            'members': members,
            'centroid': (centroid.x, centroid.y),
        })
    return islands_data


# ═══════════════════════════════════════════════════════════════════════
# 2. SAM HIERARCHICAL DETECTION  (with proper DPI scaling)
# ═══════════════════════════════════════════════════════════════════════

def run_sam_hierarchical(image, polygons, gap=3, sam_checkpoint=r"..\Models\sam_vit_b_01ec64.pth"):
    """
    Run SAM on each island with properly scaled coordinates.
    
    The polygons are in PDF space (72 DPI) but the image is rendered at
    RENDER_DPI.  Every coordinate sent to SAM must be multiplied by DPI_SCALE.
    
    Prompt strategy per island:
        • box   = scaled bounding box of the island
        • point = scaled centroid of the island (label=1, foreground)
    
    Returns:
        (final_masks, image)
        final_masks: list of dicts with 'mask', 'box_pixel', 'box_pdf', 'id'
    """
    # Setup SAM
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Build islands in PDF space
    islands = get_islands_data(polygons, gap)
    print(f"Islands detected: {len(islands)}")

    final_masks = []

    for idx, island in enumerate(islands):
        # --- Scale PDF coords → pixel coords ---
        pdf_box = np.array(island['bbox'])  # [xmin, ymin, xmax, ymax]
        pixel_box = pdf_box * DPI_SCALE

        cx_pdf, cy_pdf = island['centroid']
        pixel_point = np.array([[cx_pdf * DPI_SCALE, cy_pdf * DPI_SCALE]])
        point_label = np.array([1])  # foreground

        try:
            masks, scores, _ = predictor.predict(
                point_coords=pixel_point,
                point_labels=point_label,
                box=pixel_box,
                multimask_output=True,  # get 3 masks, pick best score
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
        for i, island in enumerate(islands_data):
            bx = np.array(island['bbox']) * DPI_SCALE
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
            # Mark centroid
            cx, cy = island['centroid']
            axes[0].plot(cx * DPI_SCALE, cy * DPI_SCALE, 'r+', markersize=6, markeredgewidth=1.5)

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

