import xml.etree.ElementTree as ET
import albumentations as A
import cv2
import os
import shutil

# 1. Parse XML bounding boxes
def parse_voc_xml(xml_path):
    """
    Parse VOC XML file and return bounding boxes and labels
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    labels = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    
    return bboxes, labels

# 2. Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
    A.Blur(blur_limit=3, p=0.3)
],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)

# 3. Apply augmentation and save results
def augment_and_save(image_path, xml_path, out_image_dir, out_xml_dir, n_aug=5):
    image_name = os.path.basename(image_path)
    
    # Check if XML path exists
    if not os.path.exists(xml_path):
        print(f"Warning: XML file not found for {image_name}, skipping.")
        return
        
    bboxes, labels = parse_voc_xml(xml_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations expects RGB
    
    for i in range(n_aug):
        try:
            augmented = transform(image=image, bboxes=bboxes, labels=labels)
            img_aug = augmented['image']
            bbox_aug = augmented['bboxes']
            labels_aug = augmented['labels']
            
            # Save augmented image
            new_image_name = f"{os.path.splitext(image_name)[0]}_aug{i}.png"
            # Convert back to BGR for saving with OpenCV
            cv2.imwrite(os.path.join(out_image_dir, new_image_name), cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
            
            # Save augmented XML
            save_voc_xml(xml_path, bbox_aug, labels_aug, os.path.join(out_xml_dir, new_image_name.replace('.png', '.xml')))
        except Exception as e:
            print(f"Error augmenting {image_name}: {e}")
            continue

# 4. Save XML function
def save_voc_xml(original_xml, bboxes, labels, out_path):
    tree = ET.parse(original_xml)
    root = tree.getroot()
    
    # Update filename in the XML to match the new augmented image name
    root.find('filename').text = os.path.basename(out_path).replace('.xml', '.png')
    
    # Remove old objects
    for obj in root.findall('object'):
        root.remove(obj)
    
    # Add new bboxes
    for bbox, label in zip(bboxes, labels):
        obj = ET.Element('object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(bbox[2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(bbox[3]))
        root.append(obj)
    
    tree.write(out_path)

if __name__ == "__main__":
    # Define your directories here
    # Example for a hypothetical 'pcb_voc' directory structure
    image_dir = "data/pcb_anomaly_detection/PCB/test/Spurious_Copper"
    xml_dir = "data/pcb_anomaly_detection/PCB/annotations/Spurious_Copper"
    out_image_dir = "data/pcb_anomaly_detection/PCB/test/aug_images/Spurious_Copper"
    out_xml_dir = "data/pcb_anomaly_detection/PCB/aug_annotations/Spurious_Copper"

    print("Starting augmentation...")
    print(f"Reading from: {image_dir}")
    print(f"Writing to:   {out_image_dir} and {out_xml_dir}")

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_xml_dir, exist_ok=True)

    image_files = [fname for fname in os.listdir(image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("\nWarning: No images found in the specified directory.")
        print("Please update the 'image_dir' and 'xml_dir' variables in the script to point to your dataset.")
    else:
        for fname in image_files:
            image_path = os.path.join(image_dir, fname)
            # Infer XML path from image path (e.g., image.png -> image.xml)
            xml_path = os.path.join(xml_dir, os.path.splitext(fname)[0] + ".xml")
            augment_and_save(image_path, xml_path, out_image_dir, out_xml_dir, n_aug=5)
        print(f"\nAugmentation complete. Generated {len(image_files) * 5} new samples.")
