import os
import json
from glob import glob
from pathlib import Path
from PIL import ImageDraw, ImageFont

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


dataset = "COCO"

#image_dir = f"/path/to/benchmark_datasets_gt/{dataset}/val/rgb"
image_dir = "/path/to/datasets/COCO/2017/val2017"
orig_gt_dir = f"/path/to/benchmark_datasets_gt/{dataset}/val/json"
label_error_dir = f"/path/to/benchmark_datasets_legt_variant1/{dataset}/val/json"
output_dir = f"/path/to/benchmark_datasets_labelerror_vis/{dataset}"

context_factor = 2.0
min_context = 80

os.makedirs(output_dir, exist_ok=True)



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_crop(box, img_w, img_h, context_factor=2.0, min_context=80):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    context_x = max(bw * context_factor, min_context)
    context_y = max(bh * context_factor, min_context)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    crop_x1 = max(0, int(cx - context_x))
    crop_y1 = max(0, int(cy - context_y))
    crop_x2 = min(img_w, int(cx + context_x))
    crop_y2 = min(img_h, int(cy + context_y))

    return crop_x1, crop_y1, crop_x2, crop_y2


def box_intersects_crop(box, crop):
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop
    return not (x2 < cx1 or x1 > cx2 or y2 < cy1 or y1 > cy2)


def shift_box_to_crop(box, crop):
    x1, y1, x2, y2 = box
    cx1, cy1, _, _ = crop
    return [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]


def draw_pil_box(draw, box, color, label=None, width=4):
    x1, y1, x2, y2 = map(int, box)

    for i in range(width):
        draw.rectangle(
            [x1 - i, y1 - i, x2 + i, y2 + i],
            outline=color,
        )

    if label is not None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()

        draw.text((x1, max(0, y1 - 28)), label, fill=color, font=font)


def draw_box(ax, box, color, label=None, linewidth=2):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)

    if label is not None:
        ax.text(
            x1,
            max(0, y1 - 4),
            label,
            color=color,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
        )


def find_image(stem):
    exts = [".jpg", ".jpeg", ".png"]
    for ext in exts:
        path = Path(image_dir) / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for {stem}")


def get_error_box(error):
    if "validated_box" in error:
        return error["validated_box"]["bbox"]
    if "original_box" in error:
        return error["original_box"]["bbox"]
    if "bbox" in error:
        return error["bbox"]
    raise KeyError("Could not find bbox in error entry.")


def get_error_label(error):
    if "validated_box" in error:
        label = error["validated_box"].get("label", "error")
        prob = error["validated_box"].get("prob", None)
        if prob is not None:
            return f"{label} ({prob:.2f})"
        return label
    return error.get("label", "error")


def get_original_error_box(error):

    possible_keys = [
        "original_box",
        "matched_original_box",
        "gt_box",
        "original_gt_box",
    ]

    for key in possible_keys:
        if key in error:
            return error[key]["bbox"], error[key].get("label", "GT")

    return None, None


error_files = sorted(glob(os.path.join(label_error_dir, "*.json")))

for error_file in error_files:
    stem = Path(error_file).stem

    orig_gt_file = Path(orig_gt_dir) / f"{stem}.json"
    image_file = find_image(stem)

    if not orig_gt_file.exists():
        print(f"Missing original GT for {stem}")
        continue

    error_data = load_json(error_file)
    orig_gt = load_json(orig_gt_file)

    img = Image.open(image_file).convert("RGB")
    img_w, img_h = img.size

    gt_objects = orig_gt.get("objects", [])
    errors = error_data.get("errors", [])

    for idx, error in enumerate(errors):
        error_type = error["type"]
        error_box = get_error_box(error)
        error_label = get_error_label(error)

        crop = compute_crop(
            error_box,
            img_w=img_w,
            img_h=img_h,
            context_factor=context_factor,
            min_context=min_context,
        )

        cropped_img = img.crop(crop)

        save_dir = Path(output_dir) / error_type
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{stem}_{idx:04d}_{error_type}.png"


        if error_type in ["missing", "misaligned"]:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cropped_img)

            for obj in gt_objects:
                gt_box = obj["bbox"]
                if box_intersects_crop(gt_box, crop):
                    shifted = shift_box_to_crop(gt_box, crop)
                    draw_box(ax, shifted, color="lime", label=obj.get("label", "GT"), linewidth=2)

            shifted_error_box = shift_box_to_crop(error_box, crop)
            draw_box(ax, shifted_error_box, color="red", label=error_label, linewidth=2.5)

            ax.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close()

  
        elif error_type in ["classification", "misclassification"]:
            left_img = cropped_img.copy()
            right_img = cropped_img.copy()

            left_draw = ImageDraw.Draw(left_img)
            right_draw = ImageDraw.Draw(right_img)

            original_box, original_label = get_original_error_box(error)

            if original_box is not None:
                shifted_original_box = shift_box_to_crop(original_box, crop)
                draw_pil_box(
                    left_draw,
                    shifted_original_box,
                    color="lime",
                    label=original_label,
                    width=4,
                )
            else:
                for obj in gt_objects:
                    gt_box = obj["bbox"]
                    if box_intersects_crop(gt_box, crop):
                        shifted = shift_box_to_crop(gt_box, crop)
                        draw_pil_box(
                            left_draw,
                            shifted,
                            color="lime",
                            label=obj.get("label", "GT"),
                            width=4,
                        )

            shifted_error_box = shift_box_to_crop(error_box, crop)
            draw_pil_box(
                right_draw,
                shifted_error_box,
                color="red",
                label=error_label,
                width=4,
            )

            w, h = cropped_img.size
            combined = Image.new("RGB", (2 * w, h))
            combined.paste(left_img, (0, 0))
            combined.paste(right_img, (w, 0))

            combined.save(save_path)

     
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cropped_img)

            for obj in gt_objects:
                gt_box = obj["bbox"]
                if box_intersects_crop(gt_box, crop):
                    shifted = shift_box_to_crop(gt_box, crop)
                    draw_box(ax, shifted, color="lime", label=obj.get("label", "GT"))

            shifted_error_box = shift_box_to_crop(error_box, crop)
            draw_box(ax, shifted_error_box, color="red", label=error_label)

            ax.set_title(f"{error_type}: {stem}")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
            plt.close()

print("Done.")