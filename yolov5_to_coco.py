import json
import os
from PIL import Image
import csv
import shutil

# Set the paths for the input and output directories
input_dir = "C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/Urchin-Detector/data"
output_dir = "C:/Users/kraw084/OneDrive - The University of Auckland/Desktop"

csv_path = "C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/Urchin-Detector/data/csvs/High_conf_dataset_V5.csv"
csv_file = open(csv_path, "r")
reader = csv.DictReader(csv_file)
csv_dataset = {int(row["id"]):row for row in reader}
csv_file.close()

txt_dir = "C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/Urchin-Detector/data/datasets/full_dataset_v5"


# Define the categories for the COCO dataset
categories = [{"id": 0, "name": "Evechinus chloroticus"},
              {"id": 1, "name": "Centrostephanus rodgersii"},
              {"id": 2, "name": "Heliocidaris erythrogramma"}]


image_dir = f"{input_dir}/images"
label_dir = f"{input_dir}/labels"

def create_annot_file(dataset):
    # Define the COCO dataset dictionary
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    f = open(txt_dir + f"/{dataset}.txt")
    acceptable_ids = f.readlines()
    extract_id = lambda x: int(x.strip("\n").split("/")[-1].split(".")[0][2:])
    acceptable_ids = list(map(extract_id, acceptable_ids))

    # Loop through the images in the input directory
    for image_file in os.listdir(image_dir):
        
        # Load the image and get its dimensions
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        width, height = image.size

        if int(image_file.split('.')[0][2:]) not in acceptable_ids:
            continue
        
        # Add the image to the COCO dataset
        image_dict = {
            "id": int(image_file.split('.')[0][2:]),
            "width": width,
            "height": height,
            "file_name": image_file
        }
        coco_dataset["images"].append(image_dict)
        
        # Load the bounding box annotations for the image
        if os.path.isfile(os.path.join(label_dir, f'{image_file.split(".")[0]}.txt')):
            with open(os.path.join(label_dir, f'{image_file.split(".")[0]}.txt')) as f:
                annotations = f.readlines()
        else:
            annotations = []
        
        # Loop through the annotations and add them to the COCO dataset
        for ann in annotations:
            x, y, w, h = map(float, ann.strip().split()[1:])
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": int(image_file.split('.')[0][2:]),
                "category_id": int(ann[0]),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)

    # Save the COCO dataset to a JSON file
    with open(os.path.join(output_dir, f'instances_{dataset}.json'), 'w') as f:
        json.dump(coco_dataset, f)


create_annot_file("train")
create_annot_file("val")
create_annot_file("test")

def create_image_dirs(dataset):
    os.makedirs(output_dir + f"/{dataset}")
    f = open(txt_dir + f"/{dataset}.txt")

    for im_file_path in f.readlines():
        im_file_path = im_file_path.strip("\n")
        shutil.copy("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/Urchin-Detector/" + im_file_path, output_dir + f"/{dataset}")


create_image_dirs("train")
create_image_dirs("val")
create_image_dirs("test")