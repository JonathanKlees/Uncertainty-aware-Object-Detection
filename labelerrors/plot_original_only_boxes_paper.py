import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def xyxy_to_patch(box, color, linewidth=2, label=None):
    x1, y1, x2, y2 = box
    return patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        label=label,
    )


def compute_crop(box, img_w, img_h, context_factor=3.0, min_context=80):

    x1, y1, x2, y2 = map(float, box)

    box_w = x2 - x1
    box_h = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    crop_w = max(box_w * context_factor, box_w + 2 * min_context)
    crop_h = max(box_h * context_factor, box_h + 2 * min_context)

    crop_size = max(crop_w, crop_h)

    crop_x1 = max(0, cx - crop_size / 2)
    crop_y1 = max(0, cy - crop_size / 2)
    crop_x2 = min(img_w, cx + crop_size / 2)
    crop_y2 = min(img_h, cy + crop_size / 2)

    return [crop_x1, crop_y1, crop_x2, crop_y2]


def box_intersects_crop(box, crop):
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop

    return not (x2 < cx1 or x1 > cx2 or y2 < cy1 or y1 > cy2)


def shift_box_to_crop(box, crop):
    x1, y1, x2, y2 = box
    cx1, cy1, _, _ = crop

    return [
        x1 - cx1,
        y1 - cy1,
        x2 - cx1,
        y2 - cy1,
    ]


def plot_original_only_for_image(
    image_path,
    vgt_json_path,
    error_json_path,
    output_dir,
    context_factor=3.0,
    min_context=100,
    show_vgt=True,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    vgt = load_json(vgt_json_path) if vgt_json_path is not None and Path(vgt_json_path).exists() else {"objects": []}
    errors = load_json(error_json_path)

    original_only_errors = [
        e for e in errors.get("errors", [])
        if e.get("type") == "original_only"
    ]

    if len(original_only_errors) == 0:
        return 0

    saved = 0

    for idx, err in enumerate(original_only_errors):
        
        if "original_box" in err:
            original_box = err["original_box"]
        elif "validated_box" in err:
            original_box = err["validated_box"]
        else:
            original_box = err

        box = original_box["bbox"]
        label = original_box.get("label", "original-only")

        crop = compute_crop(
            box,
            img_w=img_w,
            img_h=img_h,
            context_factor=context_factor,
            min_context=min_context,
        )

        crop_int = tuple(map(int, crop))
        cropped_img = img.crop(crop_int)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cropped_img)
        ax.axis("off")

        if show_vgt:
            for obj in vgt.get("objects", []):
                vgt_box = obj["bbox"]

                if not box_intersects_crop(vgt_box, crop):
                    continue

                shifted_vgt_box = shift_box_to_crop(vgt_box, crop)
                rect = xyxy_to_patch(
                    shifted_vgt_box,
                    color="lime",
                    linewidth=1.5,
                )
                ax.add_patch(rect)

                vx1, vy1, _, _ = shifted_vgt_box
                ax.text(
                    vx1,
                    max(vy1 - 3, 0),
                    obj.get("label", "VGT"),
                    color="lime",
                    fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
                )


        shifted_box = shift_box_to_crop(box, crop)
        rect = xyxy_to_patch(
            shifted_box,
            color="red",
            linewidth=3,
        )
        ax.add_patch(rect)

        x1, y1, _, _ = shifted_box
        ax.text(
            x1,
            max(y1 - 5, 0),
            f"original-only: {label}",
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, pad=2, edgecolor="none"),
        )

        out_name = f"{image_path.stem}_original_only_{idx:03d}.png"
        out_path = output_dir / out_name

        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        saved += 1

    return saved


def plot_original_only_dataset(
    image_dir,
    vgt_json_dir,
    error_json_dir,
    output_dir,
    image_suffix=".png",
    max_images=None,
):
    image_dir = Path(image_dir)
    vgt_json_dir = Path(vgt_json_dir)
    error_json_dir = Path(error_json_dir)
    output_dir = Path(output_dir)

    error_files = sorted(error_json_dir.glob("*.json"))

    total_saved = 0
    processed_images = 0

    for error_json_path in error_files:
        stem = error_json_path.stem

        image_path = image_dir / f"{stem}{image_suffix}"
        vgt_json_path = vgt_json_dir / f"{stem}.json"

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        if not vgt_json_path.exists():
            print(f"VGT json not found: {vgt_json_path}")
            vgt_json_path = None

        num_saved = plot_original_only_for_image(
            image_path=image_path,
            vgt_json_path=vgt_json_path,
            error_json_path=error_json_path,
            output_dir=output_dir,
            context_factor=3.0,
            min_context=100,
            show_vgt=True,
        )

        if num_saved > 0:
            processed_images += 1
            total_saved += num_saved
            print(f"{stem}: saved {num_saved} crops")

        if max_images is not None and processed_images >= max_images:
            break

    print(f"Done. Saved {total_saved} original-only crops.")


if __name__ == "__main__":
    dataset = "COCO"
    mode = "val"
    
    plot_original_only_dataset(
        #image_dir=f"/path/to/benchmark_datasets_gt/{dataset}/{mode}/rgb",
        image_dir=f"/path/to/datasets/COCO/2017/val2017",
        vgt_json_dir=f"/path/to/benchmark_datasets_vgt_all_bboxes/{dataset}/{mode}/json",
        error_json_dir=f"/path/to/benchmark_datasets_legt/{dataset}/{mode}/json",
        output_dir=f"/path/to/benchmark_datasets_labelerror_vis/original_only/{dataset}/{mode}",
        image_suffix=".jpg",   # change to ".jpg" if needed
        max_images=100,
    )