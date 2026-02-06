import fitz  # PyMuPDF
import cv2
import json
import numpy as np
import os

class CropEngine:
    def __init__(self, pdf_path, json_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Configuration
        self.DPI = 300
        # PyMuPDF default is 72 DPI. We need a scale factor.
        self.SCALE_FACTOR = self.DPI / 72.0 

    def process_crops(self, output_root, padding=20):
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        filename_base = os.path.splitext(self.metadata["source_file"])[0]
        
        for page_data in self.metadata["pages"]:
            page_idx = page_data["page_index"]
            objects = page_data["objects"]
            
            if not objects:
                continue
                
            print(f"Rendering page {page_idx+1} for {len(objects)} crops...")
            
            # 1. Render high-res page ONCE
            page = self.doc[page_idx]
            pix = page.get_pixmap(dpi=self.DPI)
            
            # Convert to OpenCV format (BGR)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_array.reshape(pix.h, pix.w, pix.n)
            
            # Handle RGB vs RGBA
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            img_h, img_w = img.shape[:2]

            # 2. Crop Components
            for obj in objects:
                category = obj["type"]
                vx1, vy1, vx2, vy2 = obj["bbox"]
                
                # Convert PDF Points to Image Pixels
                px1 = int(vx1 * self.SCALE_FACTOR) - padding
                py1 = int(vy1 * self.SCALE_FACTOR) - padding
                px2 = int(vx2 * self.SCALE_FACTOR) + padding
                py2 = int(vy2 * self.SCALE_FACTOR) + padding
                
                # Boundary checks
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(img_w, px2), min(img_h, py2)
                
                # Extract Region of Interest
                crop = img[py1:py2, px1:px2]
                
                if crop.size == 0:
                    continue
                
                # Save into categorized folders
                save_dir = os.path.join(output_root, category)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # Naming: File_Page_ID.png
                out_name = f"{filename_base}_p{page_idx}_id{obj['id']}.png"
                cv2.imwrite(os.path.join(save_dir, out_name), crop)
        
        print("Cropping process completed.")

# Example Usage:
# engine = CropEngine("wiring_diagram.pdf", "analysis_data.json")
# engine.process_crops("Final_Dataset")