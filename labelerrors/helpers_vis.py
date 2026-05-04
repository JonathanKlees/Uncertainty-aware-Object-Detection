import json
import os
import random
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from collections import defaultdict


def vis_main_errors(modes, rgb, img_suffix, gt, vgt, legt, class_names):
    font = ImageFont.load_default()

    for mode in modes:
        if rgb.startswith("/path/to/datasets"):
            rgb_path = rgb
        else:
            rgb_path = f"{gt}/{mode}/rgb"
            
        gt_path = f"{gt}/{mode}/json"
        vgt_path = f"{vgt}/{mode}/json"
        legt_path = f"{legt}/{mode}/json"

        vis_path = f"{legt}/{mode}/vis"
        os.makedirs(vis_path, exist_ok=True)

        legt_files = sorted(glob(f"{legt_path}/*.json"))[:100]

        for legt_file in legt_files:
            img_name = os.path.basename(legt_file).replace(".json", "")

            rgb_file = f"{rgb_path}/{img_name}{img_suffix}"
            gt_file = f"{gt_path}/{img_name}.json"
            vgt_file = f"{vgt_path}/{img_name}.json"

            if not os.path.exists(rgb_file) or not os.path.exists(gt_file):
                continue

            with open(legt_file, "r") as f:
                legt_errs = json.load(f)["errors"]

            main_errs = [e for e in legt_errs if e["type"] != "original_only"]

            if len(main_errs) == 0:
                continue

            with open(gt_file, "r") as f:
                all_gt_objs = json.load(f)["objects"]

            gt_objs = [obj for obj in all_gt_objs if obj["label"] in class_names]

            vgt_objs = []
            if os.path.exists(vgt_file):
                with open(vgt_file, "r") as f:
                    vgt_objs = json.load(f)["objects"]

            base_img = Image.open(rgb_file).convert("RGB")

            img_gt = base_img.copy()
            img_vgt = base_img.copy()
            img_legt = base_img.copy()

            draw_gt = ImageDraw.Draw(img_gt)
            draw_vgt = ImageDraw.Draw(img_vgt)
            draw_legt = ImageDraw.Draw(img_legt)

            for obj in gt_objs:
                label = obj["label"]
                bbox = obj["bbox"]
                x1, y1, x2, y2 = bbox
                w, h = font.getbbox(label)[2:]
                draw_gt.rectangle(bbox, outline="red", width=2)
                draw_gt.rectangle([x1, y1 - h, x1 + w, y1], fill="green")
                draw_gt.text((x1, y1 - h), label, fill="white", font=font)

            for obj in vgt_objs:
                label = obj["label"]
                bbox = obj["bbox"]
                x1, y1, x2, y2 = bbox
                w, h = font.getbbox(label)[2:]
                draw_vgt.rectangle(bbox, outline="green", width=2)
                draw_vgt.rectangle([x1, y1 - h, x1 + w, y1], fill="green")
                draw_vgt.text((x1, y1 - h), label, fill="white", font=font)

            for obj in main_errs:
                err_type = obj["type"]
                bbox = obj["validated_box"]["bbox"]
                x1, y1, x2, y2 = bbox
                w, h = font.getbbox(err_type)[2:]
                draw_legt.rectangle(bbox, outline="blue", width=2)
                draw_legt.rectangle([x1, y1 - h, x1 + w, y1], fill="green")
                draw_legt.text((x1, y1 - h), err_type, fill="white", font=font)

            new_img = Image.new("RGB", (base_img.width, base_img.height * 3))
            new_img.paste(img_gt, (0, 0))
            new_img.paste(img_legt, (0, base_img.height))
            new_img.paste(img_vgt, (0, base_img.height * 2))

            new_img.save(f"{vis_path}/{img_name}.png")




def vis_random_original_only_boxes(
    modes, rgb, img_suffix, gt, all_vgt, legt, class_names, num_boxes=200, seed=42
):
    font = ImageFont.load_default()
    random.seed(seed)

    all_original_only = []

    for mode in modes:
        if rgb.startswith("/path/to/datasets"):
            rgb_path = rgb
        else:
            rgb_path = f"{gt}/{mode}/rgb"
            
        gt_path = f"{gt}/{mode}/json"
        all_vgt_path = f"{all_vgt}/{mode}/json"
        legt_path = f"{legt}/{mode}/json"

        legt_files = sorted(glob(f"{legt_path}/*.json"))

        for legt_file in legt_files:
            img_name = os.path.basename(legt_file).replace(".json", "")

            rgb_file = f"{rgb_path}/{img_name}{img_suffix}"
            gt_file = f"{gt_path}/{img_name}.json"
            all_vgt_file = f"{all_vgt_path}/{img_name}.json"

            if not os.path.exists(rgb_file) or not os.path.exists(gt_file):
                continue

            with open(legt_file, "r") as f:
                legt_data = json.load(f)

            errs = legt_data.get("errors", [])
            original_only_errs = [e for e in errs if e["type"] == "original_only"]

            for err_idx, err in enumerate(original_only_errs):
                all_original_only.append({
                    "mode": mode,
                    "img_name": img_name,
                    "rgb_file": rgb_file,
                    "gt_file": gt_file,
                    "all_vgt_file": all_vgt_file,
                    "error": err,
                    "error_idx": err_idx,
                })

    if len(all_original_only) == 0:
        print("No original-only errors found.")
        return

    sample_size = min(num_boxes, len(all_original_only))
    sampled_errors = random.sample(all_original_only, sample_size)

    print(f"Total {len(all_original_only)} original only boxes")
    print(f"Visualize {sample_size} random sampled boxes.")

    vis_path = f"{legt}/vis_random_original_only"
    meta_path = f"{legt}/random_original_only_bboxes"
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(meta_path, exist_ok=True)

    grouped_boxes = defaultdict(list)

    for sample in sampled_errors:
        mode = sample["mode"]
        img_name = sample["img_name"]
        bbox = sample["error"]["original_box"]["bbox"]

        base_name = f"{mode}_{img_name}"
        grouped_boxes[base_name].append(bbox)

    for base_name, boxes in grouped_boxes.items():
        meta = {
            "original_only_boxes": boxes
        }

        meta_save_path = os.path.join(meta_path, f"{base_name}.json")
        with open(meta_save_path, "w") as f:
            json.dump(meta, f, indent=2)

    for sample_idx, sample in enumerate(sampled_errors):
        mode = sample["mode"]
        img_name = sample["img_name"]
        rgb_file = sample["rgb_file"]
        gt_file = sample["gt_file"]
        all_vgt_file = sample["all_vgt_file"]
        err = sample["error"]
        err_idx = sample["error_idx"]

        with open(gt_file, "r") as f:
            all_gt_objs = json.load(f)["objects"]

        gt_objs = [obj for obj in all_gt_objs if obj["label"] in class_names]

        all_vgt_objs = []
        if os.path.exists(all_vgt_file):
            with open(all_vgt_file, "r") as f:
                all_vgt_objs = json.load(f)["objects"]

        base_img = Image.open(rgb_file).convert("RGB")

        img_gt = base_img.copy()
        img_legt = base_img.copy()
        img_vgt = base_img.copy()

        draw_gt = ImageDraw.Draw(img_gt)
        draw_legt = ImageDraw.Draw(img_legt)
        draw_vgt = ImageDraw.Draw(img_vgt)

        for obj in gt_objs:
            label = obj["label"]
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox

            text_bbox = font.getbbox(label)
            w = text_bbox[2] - text_bbox[0]
            h = text_bbox[3] - text_bbox[1]
            y_text = max(0, y1 - h)

            draw_gt.rectangle(bbox, outline="red", width=2)
            draw_gt.rectangle([x1, y_text, x1 + w, y1], fill="green")
            draw_gt.text((x1, y_text), label, fill="white", font=font)

        bbox = err["original_box"]["bbox"]
        x1, y1, x2, y2 = bbox

        label_text = "original_only"
        text_bbox = font.getbbox(label_text)
        w = text_bbox[2] - text_bbox[0]
        h = text_bbox[3] - text_bbox[1]
        y_text = max(0, y1 - h)

        draw_legt.rectangle(bbox, outline="blue", width=2)
        draw_legt.rectangle([x1, y_text, x1 + w, y1], fill="green")
        draw_legt.text((x1, y_text), label_text, fill="white", font=font)

        for obj in all_vgt_objs:
            label = obj["label"]
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox

            text_bbox = font.getbbox(label)
            w = text_bbox[2] - text_bbox[0]
            h = text_bbox[3] - text_bbox[1]
            y_text = max(0, y1 - h)

            draw_vgt.rectangle(bbox, outline="green", width=2)
            draw_vgt.rectangle([x1, y_text, x1 + w, y1], fill="green")
            draw_vgt.text((x1, y_text), label, fill="white", font=font)

        new_img = Image.new("RGB", (base_img.width, base_img.height * 3))
        new_img.paste(img_gt, (0, 0))
        new_img.paste(img_legt, (0, base_img.height))
        new_img.paste(img_vgt, (0, base_img.height * 2))

        save_name = f"{sample_idx:03d}_{mode}_{img_name}_err{err_idx}.png"
        new_img.save(os.path.join(vis_path, save_name))
        

def vis_original_only(modes, rgb, img_suffix, gt, all_vgt, legt, class_names, max_imgs=500):
    font = ImageFont.load_default()

    for mode in modes:
        if rgb.startswith("/path/to/datasets"):
            rgb_path = rgb
        else:
            rgb_path = f"{gt}/{mode}/rgb"
            
        gt_path = f"{gt}/{mode}/json"
        all_vgt_path = f"{all_vgt}/{mode}/json"
        legt_path = f"{legt}/{mode}/json"

        vis_path = f"{legt}/{mode}/vis_original_only"
        os.makedirs(vis_path, exist_ok=True)

        legt_files = sorted(glob(f"{legt_path}/*.json"))

        saved = 0

        for legt_file in legt_files:
            if saved >= max_imgs:
                break

            img_name = os.path.basename(legt_file).replace(".json", "")

            rgb_file = f"{rgb_path}/{img_name}{img_suffix}"
            gt_file = f"{gt_path}/{img_name}.json"
            all_vgt_file = f"{all_vgt_path}/{img_name}.json"

            if not os.path.exists(rgb_file) or not os.path.exists(gt_file):
                continue

            with open(legt_file, "r") as f:
                legt_errs = json.load(f)["errors"]

            original_only_errs = [
                e for e in legt_errs if e["type"] == "original_only"
            ]

            if len(original_only_errs) == 0:
                continue

            with open(gt_file, "r") as f:
                all_gt_objs = json.load(f)["objects"]

            gt_objs = [obj for obj in all_gt_objs if obj["label"] in class_names]

            all_vgt_objs = []
            if os.path.exists(all_vgt_file):
                with open(all_vgt_file, "r") as f:
                    all_vgt_objs = json.load(f)["objects"]

            base_img = Image.open(rgb_file).convert("RGB")

            img_gt = base_img.copy()
            img_vgt = base_img.copy()
            img_legt = base_img.copy()

            draw_gt = ImageDraw.Draw(img_gt)
            draw_vgt = ImageDraw.Draw(img_vgt)
            draw_legt = ImageDraw.Draw(img_legt)

            for obj in gt_objs:
                label = obj["label"]
                bbox = obj["bbox"]
                x1, y1, x2, y2 = bbox
                w, h = font.getbbox(label)[2:]
                draw_gt.rectangle(bbox, outline="red", width=2)
                draw_gt.rectangle([x1, y1 - h, x1 + w, y1], fill="green")
                draw_gt.text((x1, y1 - h), label, fill="white", font=font)

            for obj in all_vgt_objs:
                label = obj["label"]
                bbox = obj["bbox"]
                x1, y1, x2, y2 = bbox
                w, h = font.getbbox(label)[2:]
                draw_vgt.rectangle(bbox, outline="green", width=2)
                draw_vgt.rectangle([x1, y1 - h, x1 + w, y1], fill="green")
                draw_vgt.text((x1, y1 - h), label, fill="white", font=font)

            for obj in original_only_errs:
                bbox = obj["original_box"]["bbox"]
                x1, y1, x2, y2 = bbox
                w, h = font.getbbox("original_only")[2:]
                draw_legt.rectangle(bbox, outline="blue", width=2)
                draw_legt.rectangle([x1, y1 - h, x1 + w, y1], fill="green")
                draw_legt.text((x1, y1 - h), "original_only", fill="white", font=font)

            new_img = Image.new("RGB", (base_img.width, base_img.height * 3))
            new_img.paste(img_gt, (0, 0))
            new_img.paste(img_legt, (0, base_img.height))
            new_img.paste(img_vgt, (0, base_img.height * 2))

            new_img.save(f"{vis_path}/{img_name}.png")
            saved += 1