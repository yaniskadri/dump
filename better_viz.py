import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import cv2


def visualize_sam_comprehensive(image, sam_results, polygons=None, islands_data=None, save_path=None):
    """
    Comprehensive visualization showing:
    - Original image
    - All bounding boxes
    - SAM detected masks with labels
    - Original polygons (optional)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # 1. Original Image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Bounding Boxes Only
    axes[0, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for idx, res in enumerate(sam_results):
        box = res['box']
        rect = mpatches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        axes[0, 1].add_patch(rect)
        # Add label
        axes[0, 1].text(
            box[0], box[1] - 5, 
            f'Island {idx}', 
            color='red', 
            fontsize=10, 
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    axes[0, 1].set_title(f'Bounding Boxes ({len(sam_results)} islands)', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. SAM Masks with Labels
    axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    overlay = np.zeros_like(image, dtype=np.float32)
    
    for idx, res in enumerate(sam_results):
        # Generate consistent color for this island
        np.random.seed(idx)  # Consistent colors
        color = np.random.randint(50, 255, (3,)).astype(np.float32)
        
        mask = res['mask']
        # Apply colored overlay
        overlay[mask] = color
        
        # Draw bounding box
        box = res['box']
        rect = mpatches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=2, 
            edgecolor='white', 
            facecolor='none',
            linestyle='--'
        )
        axes[1, 0].add_patch(rect)
        
        # Calculate mask centroid for label
        mask_points = np.argwhere(mask)
        if len(mask_points) > 0:
            centroid_y, centroid_x = mask_points.mean(axis=0)
            axes[1, 0].text(
                centroid_x, centroid_y, 
                f'{idx}', 
                color='white', 
                fontsize=14, 
                fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.8)
            )
    
    # Blend overlay with original image
    blended = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) * 0.5 + overlay * 0.5
    axes[1, 0].imshow(blended.astype(np.uint8))
    axes[1, 0].set_title(f'SAM Detections ({len(sam_results)} objects)', fontsize=16, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. Detailed View: Masks + Prompts Visualization
    axes[1, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if islands_data:
        for idx, island in enumerate(islands_data):
            # Draw parent polygon (positive prompt)
            parent = island['parent']
            parent_coords = np.array(parent.exterior.coords)
            axes[1, 1].plot(parent_coords[:, 0], parent_coords[:, 1], 
                           'g-', linewidth=2, label='Parent' if idx == 0 else '')
            axes[1, 1].plot(parent.centroid.x, parent.centroid.y, 
                           'g*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
            
            # Draw children polygons (negative prompts)
            for child_idx, child in enumerate(island['children'][:10]):
                child_coords = np.array(child.exterior.coords)
                axes[1, 1].plot(child_coords[:, 0], child_coords[:, 1], 
                               'r-', linewidth=1, alpha=0.5, 
                               label='Children' if idx == 0 and child_idx == 0 else '')
                axes[1, 1].plot(child.centroid.x, child.centroid.y, 
                               'rx', markersize=10, markeredgewidth=2)
            
            # Draw bounding box
            box = island['bbox']
            rect = mpatches.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1],
                linewidth=2, 
                edgecolor='cyan', 
                facecolor='none',
                linestyle=':'
            )
            axes[1, 1].add_patch(rect)
    
    axes[1, 1].set_title('SAM Prompts: Green*=Positive, Red×=Negative', fontsize=16, fontweight='bold')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def visualize_individual_islands(image, sam_results, islands_data, save_dir=None):
    """
    Create individual plots for each island to inspect them closely
    """
    n_islands = len(sam_results)
    cols = min(4, n_islands)
    rows = (n_islands + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_islands == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (res, island) in enumerate(zip(sam_results, islands_data)):
        ax = axes[idx]
        
        # Crop image to bounding box with padding
        box = res['box']
        padding = 20
        x1 = max(0, int(box[0]) - padding)
        y1 = max(0, int(box[1]) - padding)
        x2 = min(image.shape[1], int(box[2]) + padding)
        y2 = min(image.shape[0], int(box[3]) + padding)
        
        cropped_img = image[y1:y2, x1:x2]
        cropped_mask = res['mask'][y1:y2, x1:x2]
        
        # Show image with mask overlay
        overlay = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB).copy()
        overlay[cropped_mask] = overlay[cropped_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        ax.imshow(overlay)
        
        # Draw parent and children in cropped coords
        parent = island['parent']
        parent_coords = np.array(parent.exterior.coords)
        ax.plot(parent_coords[:, 0] - x1, parent_coords[:, 1] - y1, 
               'b-', linewidth=2, label='Parent')
        ax.plot(parent.centroid.x - x1, parent.centroid.y - y1, 
               'b*', markersize=15)
        
        for child in island['children'][:10]:
            child_coords = np.array(child.exterior.coords)
            ax.plot(child_coords[:, 0] - x1, child_coords[:, 1] - y1, 
                   'r-', linewidth=1, alpha=0.7)
            ax.plot(child.centroid.x - x1, child.centroid.y - y1, 
                   'rx', markersize=8)
        
        # Stats
        mask_area = cropped_mask.sum()
        parent_area = island['parent'].area
        n_children = len(island['children'])
        
        ax.set_title(f'Island {idx}\nMask: {mask_area:.0f}px | Parent: {parent_area:.0f} | Children: {n_children}', 
                    fontsize=10)
        ax.legend(fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_islands, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = f"{save_dir}/individual_islands.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Individual islands saved to {save_path}")
    
    plt.show()


def print_diagnostic_info(sam_results, islands_data, polygons):
    """
    Print detailed diagnostic information about the detection
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC INFORMATION")
    print("="*80)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  - Total polygons extracted: {len(polygons)}")
    print(f"  - Islands detected: {len(islands_data)}")
    print(f"  - SAM masks generated: {len(sam_results)}")
    
    print(f"\n🏝️  Island Details:")
    for idx, island in enumerate(islands_data):
        box = island['bbox']
        parent_area = island['parent'].area
        n_children = len(island['children'])
        
        print(f"\n  Island {idx}:")
        print(f"    BBox: ({box[0]:.1f}, {box[1]:.1f}) → ({box[2]:.1f}, {box[3]:.1f})")
        print(f"    Size: {box[2]-box[0]:.1f} × {box[3]-box[1]:.1f}")
        print(f"    Parent area: {parent_area:.1f}")
        print(f"    Children: {n_children}")
        if n_children > 0:
            child_areas = [c.area for c in island['children']]
            print(f"    Children areas: min={min(child_areas):.1f}, max={max(child_areas):.1f}, avg={np.mean(child_areas):.1f}")
    
    print(f"\n🎭 SAM Results:")
    for idx, res in enumerate(sam_results):
        mask_area = res['mask'].sum()
        box = res['box']
        print(f"  Mask {idx}: {mask_area:,} pixels covered")
    
    print("\n" + "="*80)