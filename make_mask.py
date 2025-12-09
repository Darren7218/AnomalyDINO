"""
Generate binary ground-truth masks from annotations for a dataset of PCB images.

Supported annotation formats:
 - Pascal VOC XML with <object>/<bndbox> (xmin,ymin,xmax,ymax)  -> rectangle mask
 - Pascal/VGG XML with <polygon>/<pt>/<x>/<y> (common format variants) -> polygon mask
 - COCO JSON (polygons / RLE) if pycocotools is installed
 - Existing mask images (optional) : if mask_path pattern matches image, it will be normalized and copied.

Output:
 - For each image in `images_dir` this creates labels_out/<image_basename>_mask.png with size = image size,
   single channel uint8 with 0 for background and 255 for object pixels (binary mask).
"""

import os
import sys
import json
import glob
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import cv2

# Optional COCO support
try:
    from pycocotools import mask as mask_utils
    COCO_AVAILABLE = True
except Exception:
    COCO_AVAILABLE = False


# ---------- CONFIG ----------
images_dir = "data/pcb_anomaly_detection/pcb/test/Missing_hole"            # folder with your images (jpg/png)
voc_xml_dir = "data/pcb_anomaly_detection/pcb/Annotations/Missing_hole"       # folder with Pascal VOC XMLs (same basename as images)
coco_json = None                  # path to COCO json (or None)
existing_mask_dir = None          # folder with pre-made masks (if any); will copy/normalize if present
output_dir = "data/pcb_anomaly_detection/pcb/ground_truth/Missing_hole"        # where generated masks will be saved
binary_output = True              # True -> output 0/255 mask. If False -> multi-class values kept (1..N)
class_map = None                  # Optional dict mapping class name -> integer id (for multi-class). None => everything = 1
overwrite = True                  # overwrite existing masks
# -----------------------------

os.makedirs(output_dir, exist_ok=True)

# Helper to write mask
def save_mask(mask, out_path, binary=True):
    """
    mask: 2D numpy array of integers (0 for background, >0 for classes/instances)
    binary: if True convert any non-zero to 255; else save mask* (uint8)
    """
    if binary:
        out = (mask > 0).astype(np.uint8) * 255
    else:
        # ensure values fit into 0..255
        out = np.clip(mask, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, out)

# Parse polygon pts commonly present in some XML exports
def parse_polygon_object(obj_elem):
    """
    try to extract polygon points from <object> element in XML.
    supports patterns like:
      <polygon><pt><x>...</x><y>...</y></pt>...</polygon>
    returns list of [ [x1,y1], [x2,y2], ... ] or None
    """
    poly = obj_elem.find('polygon')
    if poly is None:
        # Some tools use <segmentation> <points> or <segm> etc. Check common names:
        poly = obj_elem.find('segmentation') or obj_elem.find('segm') or obj_elem.find('shape')
    if poly is None:
        return None

    pts = []
    # two common layouts: <pt><x>..</x><y>..</y></pt>  OR  <point>x,y x,y ...</point>
    for pt in poly.findall('pt'):
        x = pt.find('x')
        y = pt.find('y')
        if x is None or y is None:
            continue
        pts.append([float(x.text), float(y.text)])
    if pts:
        return pts

    # try if polygon uses <point> text with pairs
    text = poly.text
    if text:
        # try split whitespace or commas
        coords = []
        for token in text.strip().replace(',', ' ').split():
            try:
                coords.append(float(token))
            except:
                pass
        if len(coords) >= 6 and len(coords) % 2 == 0:
            pts = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
            return pts

    return None

# Create mask from a single VOC XML file
def xml_to_mask(xml_path, img_shape, class_map=None, binary=True):
    """
    xml_path: path to Pascal VOC XML
    img_shape: (h, w) or (h,w,channels)
    returns 2D numpy array mask with 0 background and integer class ids for objects
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    h = int(img_shape[0])
    w = int(img_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)

    for obj in root.findall('object'):
        name_elem = obj.find('name')
        cls_name = name_elem.text.strip() if name_elem is not None else 'obj'
        cid = 1
        if class_map and cls_name in class_map:
            cid = int(class_map[cls_name])
        elif class_map and cls_name not in class_map:
            # skip unknown classes if class_map restricts
            continue

        # first try polygon
        poly_pts = parse_polygon_object(obj)
        if poly_pts:
            pts = np.array(poly_pts, dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], color=cid)
            continue

        # else try bndbox
        bnd = obj.find('bndbox')
        if bnd is not None:
            try:
                xmin = int(float(bnd.find('xmin').text))
                ymin = int(float(bnd.find('ymin').text))
                xmax = int(float(bnd.find('xmax').text))
                ymax = int(float(bnd.find('ymax').text))
            except Exception:
                continue
            # clamp coords
            xmin = max(0, min(w-1, xmin))
            ymin = max(0, min(h-1, ymin))
            xmax = max(0, min(w-1, xmax))
            ymax = max(0, min(h-1, ymax))
            if xmax > xmin and ymax > ymin:
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color=cid, thickness=-1)
            continue

        # else nothing recognized — skip

    return mask

# Create mask from COCO JSON
def coco_to_masks(coco_json_path, images_dir, out_dir, class_map=None, binary=True):
    if not COCO_AVAILABLE:
        print("pycocotools not available, skipping COCO conversion.")
        return
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images_info = {img['id']: img for img in coco.get('images', [])}
    anns = coco.get('annotations', [])

    # group annotations by image id
    anns_per_image = {}
    for a in anns:
        anns_per_image.setdefault(a['image_id'], []).append(a)

    for img_id, img_info in images_info.items():
        fname = img_info['file_name']
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}, skipping")
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for a in anns_per_image.get(img_id, []):
            cid = 1
            if class_map:
                # map coco category_id to desired id if provided as dict {category_name: id}
                # else use annotation['category_id']
                pass
            # annotation segmentation can be polygon(s) or RLE
            segm = a.get('segmentation', None)
            if segm is None:
                # try bbox as fallback
                bbox = a.get('bbox', None)
                if bbox:
                    x, y, bw, bh = bbox
                    x1 = int(round(x)); y1 = int(round(y))
                    x2 = int(round(x + bw)); y2 = int(round(y + bh))
                    cv2.rectangle(mask, (x1, y1), (x2, y2), color=cid, thickness=-1)
                continue

            # polygons: list of lists
            if isinstance(segm, list):
                for poly in segm:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    pts_i = np.round(pts).astype(np.int32)
                    cv2.fillPoly(mask, [pts_i], color=cid)
            else:
                # RLE
                rle = segm
                if isinstance(rle, dict):
                    # convert to mask via pycocotools
                    m = mask_utils.decode(rle)
                    if m.ndim == 3:
                        m = m.any(axis=2).astype(np.uint8)
                    mask[m > 0] = cid

        out_path = os.path.join(out_dir, Path(fname).stem + "_mask.png")
        save_mask(mask, out_path, binary=binary)
        print("Wrote", out_path)


# Main batch processing for VOC-style xml + images
def batch_create_masks(images_dir, xml_dir, out_dir, existing_mask_dir=None,
                       class_map=None, binary=True, overwrite=False, coco_json_path=None):
    # If COCO provided, process and return
    if coco_json_path:
        if COCO_AVAILABLE:
            coco_to_masks(coco_json_path, images_dir, out_dir, class_map=class_map, binary=binary)
        else:
            print("COCO JSON provided but pycocotools not found — skipping COCO.")
        return

    # build list of image files
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(images_dir, e)))

    if not image_paths:
        print("No images found in", images_dir)
        return

    for img_path in image_paths:
        base = Path(img_path).stem
        out_path = os.path.join(out_dir, f"{base}_mask.png")
        if os.path.exists(out_path) and not overwrite:
            print("Skipping (exists):", out_path)
            continue

        # if existing mask dir provided and mask exists, copy/normalize it
        if existing_mask_dir:
            candidate = os.path.join(existing_mask_dir, base + ".png")
            if os.path.exists(candidate):
                m = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    print("Failed to read existing mask:", candidate)
                else:
                    # normalize to binary
                    save_mask((m > 0).astype(np.uint8), out_path, binary=binary)
                    print("Copied existing mask:", out_path)
                    continue

        # try to find xml annotation with same basename
        xml_path = os.path.join(xml_dir, base + ".xml")
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to read image", img_path)
            continue
        h, w = img.shape[:2]

        if os.path.exists(xml_path):
            mask = xml_to_mask(xml_path, (h, w), class_map=class_map, binary=binary)
            save_mask(mask, out_path, binary=binary)
            print("Wrote", out_path)
        else:
            # no annotation found — write empty mask (or skip)
            print("No XML for", img_path, "- writing empty mask.")
            empty = np.zeros((h, w), dtype=np.uint8)
            save_mask(empty, out_path, binary=binary)

# Run script
if __name__ == "__main__":
    # Quick config override via CLI args (optional)
    # e.g. python make_masks.py /path/images /path/annots output_dir
    if len(sys.argv) >= 4:
        images_dir = sys.argv[1]
        voc_xml_dir = sys.argv[2]
        output_dir = sys.argv[3]
        os.makedirs(output_dir, exist_ok=True)

    batch_create_masks(images_dir, voc_xml_dir, output_dir,
                       existing_mask_dir=existing_mask_dir,
                       class_map=class_map,
                       binary=binary_output,
                       overwrite=overwrite,
                       coco_json_path=coco_json)
