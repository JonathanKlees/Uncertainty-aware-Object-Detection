import json
import numpy as np

probability_thresh = 0.5
softlabel_json_path = "/path/to/benchmark_paper_post_processing/our_datasets"
save_json_path = "/path/to/benchmark_datasets_vgt_all_bboxes"
gt_json_path = "/path/to/benchmark_datasets_gt"


def Cityscapes():
    softlabel_json_files = ["soft_Cityscapes_train.json", "soft_Cityscapes_val.json"]
    
    for softlabel_json_file in softlabel_json_files:
        mode = softlabel_json_file.split("_")[-1].replace(".json", "")
        
        vgt_data = json.load(open(f"{softlabel_json_path}/{softlabel_json_file}"))

        classes = vgt_data["classes"]
        objects = vgt_data["objects"]
        
        for img_name, objs in objects.items():
            print(img_name)
            
            gt_json = json.load(open(f"{gt_json_path}/Cityscapes/{mode}/json/{img_name}.json"))
            
            height, width = gt_json["imgHeight"], gt_json["imgWidth"]
                
            objs_dicti = []
            
            for obj in objs:
                bbox = obj["bbox"]
                postprocessed_softlabel = obj["postprocessed_soft_label"]
                
                clas = int(np.argmax(postprocessed_softlabel))
                class_name = classes[clas]
                prob = postprocessed_softlabel[clas]
              
                objs_dicti.append({
                    "bbox": bbox,
                    "postprocessed_soft_label": postprocessed_softlabel,
                    "label": class_name,
                    "prob": prob
                })

            img_dicti = {
                "imgHeight": height,
                "imgWidth": width,
                "objects": objs_dicti
            }
            
            with open(f"{save_json_path}/Cityscapes/{mode}/json/{img_name}.json", "w") as f:
                json.dump(img_dicti, f, indent=4) 
                
    
def PascalVOC():
    """get data split"""
    train_val_split = json.load(open("/path/to/benchmark_paper_post_processing/our_datasets/splits/PascalVOC_train_val_split.json"))
    train_imgs = train_val_split["train_imgs"]
    val_imgs = train_val_split["val_imgs"]
    
    """save hard labels"""
    softlabel_json_files = ["soft_PascalVOC_2012_detection_trainval.json"]
    
    for softlabel_json_file in softlabel_json_files:
        vgt_data = json.load(open(f"{softlabel_json_path}/{softlabel_json_file}"))

        categories = vgt_data["categories"]

        classes = []
        for category in categories:
            classes.append(category["name"])
            
        objects = vgt_data["objects"]
        
        for img_name, objs in objects.items():
            img_name = img_name.replace(".jpg","")
            print(img_name)
            
            if img_name in train_imgs:
                mode = "train"
            elif img_name in val_imgs:
                mode = "val"
            
            gt_json = json.load(open(f"{gt_json_path}/PascalVOC/{mode}/json/{img_name}.json"))
            
            height, width = gt_json["imgHeight"], gt_json["imgWidth"]
                
            objs_dicti = []
            
            for obj in objs:
                bbox = obj["bbox"]
                postprocessed_softlabel = obj["postprocessed_soft_label"]
                
                clas = int(np.argmax(postprocessed_softlabel))
                class_name = classes[clas]
                prob = postprocessed_softlabel[clas]
              
                objs_dicti.append({
                    "bbox": bbox,
                    "soft_label": postprocessed_softlabel,
                    "label": class_name,
                    "prob": prob
                })

            img_dicti = {
                "height": height,
                "width": width,
                "objects": objs_dicti
            }
            
            with open(f"{save_json_path}/PascalVOC/{mode}/json/{img_name}.json", "w") as f:
                json.dump(img_dicti, f, indent=4) 

    
def PascalVOCSegmentation():
    softlabel_json_files = ["soft_PascalVOC_2012_segmentation_train.json", "soft_PascalVOC_2012_segmentation_val.json"]
    
    for softlabel_json_file in softlabel_json_files:
        mode = softlabel_json_file.split("_")[-1].replace(".json", "")
        
        vgt_data = json.load(open(f"{softlabel_json_path}/{softlabel_json_file}"))

        classes = vgt_data["classes"]
        objects = vgt_data["objects"]
        
        for img_name, objs in objects.items():
            img_name = img_name.replace(".jpg","")
            print(img_name)
                
            objs_dicti = []
            
            for obj in objs:
                bbox = obj["bbox"]
                postprocessed_softlabel = obj["postprocessed_soft_label"]
                
                clas = int(np.argmax(postprocessed_softlabel))
                class_name = classes[clas]
                prob = postprocessed_softlabel[clas]
              
                objs_dicti.append({
                    "bbox": bbox,
                    "postprocessed_soft_label": postprocessed_softlabel,
                    "label": class_name,
                    "prob": prob
                })

            img_dicti = {
                "objects": objs_dicti
            }
            
            with open(f"{save_json_path}/PascalVOCSegmentation/{mode}/json/{img_name}.json", "w") as f:
                json.dump(img_dicti, f, indent=4) 
    
def Kitti():
    """get data split"""
    train_val_split = json.load(open("/path/to/benchmark_paper_post_processing/our_datasets/splits/Kitti_train_val_split.json"))
    train_imgs = train_val_split["train_imgs"]
    val_imgs = train_val_split["val_imgs"]
    
    """save hard labels"""
    softlabel_json_files = ["soft_Kitti_2D_train.json"]
    
    for softlabel_json_file in softlabel_json_files:
        vgt_data = json.load(open(f"{softlabel_json_path}/{softlabel_json_file}"))

        classes = vgt_data["classes"]
        objects = vgt_data["objects"]
        
        for img_name, objs in objects.items():
            print(img_name)
            
            if img_name in train_imgs:
                mode = "train"
            elif img_name in val_imgs:
                mode = "val"
            
            gt_json = json.load(open(f"{gt_json_path}/Kitti/{mode}/json/{img_name}.json"))
            
            height, width = gt_json["height"], gt_json["width"]
                
            objs_dicti = []
            
            for obj in objs:
                bbox = obj["bbox"]
                postprocessed_softlabel = obj["postprocessed_soft_label"]
                
                clas = int(np.argmax(postprocessed_softlabel))
                class_name = classes[clas]
                prob = postprocessed_softlabel[clas]
              
                objs_dicti.append({
                    "bbox": bbox,
                    "postprocessed_soft_label": postprocessed_softlabel,
                    "label": class_name,
                    "prob": prob
                })

            img_dicti = {
                "height": height,
                "width": width,
                "objects": objs_dicti
            }
            
            with open(f"{save_json_path}/Kitti/{mode}/json/{img_name}.json", "w") as f:
                json.dump(img_dicti, f, indent=4) 
        

def COCO():
    softlabel_json_files = ["soft_COCO_2017_val.json"]
    
    for softlabel_json_file in softlabel_json_files:
        vgt_data = json.load(open(f"{softlabel_json_path}/{softlabel_json_file}"))

        classes = vgt_data["classes"]
        objects = vgt_data["objects"]
        
        for img_name, objs in objects.items():
            img_name = img_name.replace(".jpg","")
            print(img_name)
            
            mode = "val"
            
            gt_json = json.load(open(f"{gt_json_path}/COCO/{mode}/json/{img_name}.json"))
            
            height, width = gt_json["imgHeight"], gt_json["imgWidth"]
                
            objs_dicti = []
            
            for obj in objs:
                bbox = obj["bbox"]
                postprocessed_softlabel = obj["postprocessed_soft_label"]
                
                clas = int(np.argmax(postprocessed_softlabel))
                class_name = classes[clas]
                prob = postprocessed_softlabel[clas]
              
                objs_dicti.append({
                    "bbox": bbox,
                    "postprocessed_soft_label": postprocessed_softlabel,
                    "label": class_name,
                    "prob": prob
                })

            img_dicti = {
                "height": height,
                "width": width,
                "objects": objs_dicti
            }
            
            with open(f"{save_json_path}/COCO/{mode}/json/{img_name}.json", "w") as f:
                json.dump(img_dicti, f, indent=4) 


if __name__ == "__main__":
    # datasets: Cityscapes, Kitti, COCO, PascalVOC, PascalVOCSegmentation
    dataset = "COCO"
   
    if dataset == "Cityscapes":
        Cityscapes()
    elif dataset == "PascalVOC":
        PascalVOC()
    elif dataset == "PascalVOCSegmentation":
        PascalVOCSegmentation()
    elif dataset == "Kitti":
        Kitti()
    elif dataset == "COCO":
        COCO()
