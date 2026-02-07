import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2


def visualize_sam_boxes_only(image, sam_results, save_path=None, figsize=(20, 15)):
    """
    MINIMAL visualization: Only bounding boxes with tiny labels.
    NO masks, NO overlay - you can see the original image clearly.
    
    Args:
        image: BGR image array
        sam_results: List of dicts with 'mask', 'box', 'id' keys
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Show original image WITHOUT any overlay
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    
    # Draw ONLY bounding boxes with tiny labels
    for idx, res in enumerate(sam_results):
        # Generate color
        np.random.seed(idx * 42)
        color = np.random.rand(3,)  # Random color for each box
        
        box = res['box']
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Draw thin bounding box
        rect = mpatches.Rectangle(
            (box[0], box[1]), 
            width, 
            height,
            linewidth=2, 
            edgecolor=color,
            facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)
        
        # Add TINY label at top-left corner
        ax.text(
            box[0] + 2, box[1] - 2,  # Just outside the box
            f'{idx}',
            color='white',
            fontsize=6,  # VERY SMALL
            fontweight='bold',
            bbox=dict(
                boxstyle='square,pad=0.2',
                facecolor=color,
                edgecolor='none',
                alpha=0.8
            )
        )
    
    ax.set_title(f'SAM Detections: {len(sam_results)} objects', 
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()
    
    # Print what was detected
    print(f"\n{'='*60}")
    print(f"DETECTED OBJECTS:")
    for idx, res in enumerate(sam_results):
        box = res['box']
        w = box[2] - box[0]
        h = box[3] - box[1]
        print(f"  [{idx}] BBox: ({box[0]:.0f},{box[1]:.0f}) size:{w:.0f}×{h:.0f}")
    print(f"{'='*60}\n")


def check_why_only_one_island(islands_data, sam_results):
    """
    Debug function to check if the problem is in island creation or SAM iteration.
    """
    print("\n" + "!"*80)
    print("DEBUGGING: WHY ONLY ONE ISLAND?")
    print("!"*80)
    
    print(f"\n1. Islands created: {len(islands_data)}")
    if islands_data:
        for idx, island in enumerate(islands_data):
            print(f"   Island {idx}: bbox={island['bbox']}, children={len(island['children'])}")
    else:
        print("   ❌ NO ISLANDS CREATED! Problem is in get_islands_data()")
    
    print(f"\n2. SAM results returned: {len(sam_results)}")
    if sam_results:
        for idx, res in enumerate(sam_results):
            print(f"   Result {idx}: id={res.get('id', 'N/A')}, box={res['box']}")
    else:
        print("   ❌ NO SAM RESULTS! Problem is in run_sam_hierarchical()")
    
    print(f"\n3. Analysis:")
    if len(islands_data) > 1 and len(sam_results) == 1:
        print("   ❌ PROBLEM: Multiple islands but only 1 SAM result")
        print("   → The loop in run_sam_hierarchical() is breaking early")
        print("   → OR exceptions are being caught silently")
        print("   → Check for 'continue' or 'break' statements in the loop")
    elif len(islands_data) == 1 and len(sam_results) == 1:
        print("   ✓ Only 1 island was created, so 1 result is correct")
        print("   → Problem is in get_islands_data() - it's merging everything")
    elif len(islands_data) > 1 and len(sam_results) > 1:
        print("   ✓ Multiple islands and multiple results - looks OK")
    
    print("!"*80 + "\n")


# Minimal main script
if __name__ == "__main__":
    """
    Usage:
    
    from minimal_viz import visualize_sam_boxes_only, check_why_only_one_island
    
    # After running SAM
    masks, _ = run_sam_hierarchical(image, polygons)
    
    # Debug first
    check_why_only_one_island(islands_data, masks)
    
    # Visualize
    visualize_sam_boxes_only(image, masks, save_path="boxes_only.png")
    """
    pass
