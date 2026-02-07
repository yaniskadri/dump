import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2


def visualize_sam_simple(image, sam_results, save_path=None, figsize=(20, 15)):
    """
    Simple single-image visualization showing SAM detected objects with bounding boxes and labels.
    
    Args:
        image: BGR image array
        sam_results: List of dicts with 'mask', 'box', 'id' keys
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Convert to RGB and create colored overlay
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy().astype(np.float32)
    
    # Process each SAM detection
    for idx, res in enumerate(sam_results):
        # Generate consistent color for this detection
        np.random.seed(idx * 42)  # Consistent colors across runs
        color = np.random.randint(80, 255, (3,)).astype(np.float32)
        
        mask = res['mask']
        
        # Apply colored overlay where mask is True
        overlay[mask] = overlay[mask] * 0.4 + color * 0.6
        
        # Draw bounding box
        box = res['box']
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        rect = mpatches.Rectangle(
            (box[0], box[1]), 
            width, 
            height,
            linewidth=3, 
            edgecolor=color / 255.0,  # Normalize to 0-1 for matplotlib
            facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)
        
        # Add small label at top-left of bounding box
        label_text = f"#{idx}"
        
        # Add background box for better readability
        ax.text(
            box[0] + 5, box[1] + 15,  # Slight offset from corner
            label_text,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor=color / 255.0,
                edgecolor='white',
                linewidth=1.5,
                alpha=0.9
            )
        )
        
        # Optionally add area info
        if 'mask_area' in res:
            area_text = f"{res['mask_area']:,}px"
            ax.text(
                box[0] + 5, box[1] + height - 5,
                area_text,
                color='white',
                fontsize=8,
                verticalalignment='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='black',
                    alpha=0.7
                )
            )
    
    # Display the overlay
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title(f'SAM Detections: {len(sam_results)} objects found', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SAM DETECTION SUMMARY")
    print(f"{'='*60}")
    for idx, res in enumerate(sam_results):
        mask_area = res.get('mask_area', res['mask'].sum())
        box = res['box']
        width = box[2] - box[0]
        height = box[3] - box[1]
        print(f"Object #{idx}: {width:.0f}×{height:.0f}px bbox, {mask_area:,}px mask area")
    print(f"{'='*60}\n")


def debug_mask_coverage(sam_results):
    """
    Debug function to check why masks might not be showing.
    """
    print("\n🔍 DEBUGGING MASK COVERAGE:")
    print("="*60)
    
    for idx, res in enumerate(sam_results):
        mask = res['mask']
        box = res['box']
        
        # Check mask properties
        mask_pixels = mask.sum()
        total_pixels = mask.size
        coverage = (mask_pixels / total_pixels) * 100
        
        # Check if mask is within box
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height
        
        print(f"\nObject #{idx}:")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask pixels: {mask_pixels:,} / {total_pixels:,} ({coverage:.2f}%)")
        print(f"  BBox: ({box[0]:.0f}, {box[1]:.0f}) -> ({box[2]:.0f}, {box[3]:.0f})")
        print(f"  BBox area: {box_area:.0f}px²")
        print(f"  Mask/BBox ratio: {(mask_pixels/box_area)*100:.1f}%")
        
        # Check for common issues
        if mask_pixels == 0:
            print(f"  ⚠️  WARNING: Mask is completely empty!")
        elif mask_pixels < 100:
            print(f"  ⚠️  WARNING: Very small mask (< 100 pixels)")
        elif coverage < 0.01:
            print(f"  ⚠️  WARNING: Mask covers < 0.01% of image")
        else:
            print(f"  ✓ Mask appears valid")
    
    print("="*60)


# Usage example
if __name__ == "__main__":
    """
    Example usage in your main script:
    
    from simple_sam_visualization import visualize_sam_simple, debug_mask_coverage
    
    # After running SAM
    masks, _ = run_sam_hierarchical(image, polygons)
    
    # Debug first to check for issues
    debug_mask_coverage(masks)
    
    # Then visualize
    visualize_sam_simple(image, masks, save_path="sam_detections.png")
    """
    pass
