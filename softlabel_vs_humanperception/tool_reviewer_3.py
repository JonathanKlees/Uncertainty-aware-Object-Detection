
import json
import os
import getch
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------
# Settings
# --------------------------------------------------

DATASET_NAME = "PascalVOC"
REVIEWER_ID = "reviewer_3"

INPUT_JSON = f"samples/{DATASET_NAME}.json"
OUTPUT_JSON = f"review_decisions/{DATASET_NAME}_class_or_not_review_{REVIEWER_ID}.json"

#IMAGE_DIR = f"/path/to/benchmark_rgb_softlabelprop_vs_human_perception/{DATASET_NAME}"
#IMAGE_DIR = f"/path/to/datasets/COCO/2017/val2017"
IMAGE_DIR = f"/path/to/VOC2012_train_val/JPEGImages"

# rendered preview image
REVIEW_IMAGE_DIR = "review_images"

ANSWER_KEY = "human_class_agreement"

KEY_TO_SCORE = {
    "^": 0.0,
    "1": 0.1,
    "2": 0.2,
    "3": 0.3,
    "4": 0.4,
    "5": 0.5,
    "6": 0.6,
    "7": 0.7,
    "8": 0.8,
    "9": 0.9,
    "0": 1.0,
}

os.makedirs(REVIEW_IMAGE_DIR, exist_ok=True)


# ==========================================================
# HELPERS
# ==========================================================

def get_rendered_image_path():
    return os.path.join(REVIEW_IMAGE_DIR, f"current_{REVIEWER_ID}.jpg")


def find_image_path(image_dir, image_name):
    candidate = os.path.join(image_dir, image_name)
    if os.path.exists(candidate):
        return candidate

    for root, _, files in os.walk(image_dir):
        if image_name in files:
            return os.path.join(root, image_name)

    raise FileNotFoundError(f"Could not find image '{image_name}' in '{image_dir}'")


def save_state(file_dict):
    output_dir = os.path.dirname(OUTPUT_JSON)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(file_dict, f, indent=4)


def ensure_answer_key_exists(samples):
    for sample in samples:
        if ANSWER_KEY not in sample:
            sample[ANSWER_KEY] = {}
        elif not isinstance(sample[ANSWER_KEY], dict):
            raise TypeError(
                f"Sample {sample.get('box_id', '<unknown>')} has non-dict value for {ANSWER_KEY}"
            )


def get_resume_index(samples):
    for i, sample in enumerate(samples):
        if REVIEWER_ID not in sample.get(ANSWER_KEY, {}):
            return i
    return len(samples)


def count_reviewed_by_current_reviewer(samples):
    return sum(
        REVIEWER_ID in sample.get(ANSWER_KEY, {})
        for sample in samples
    )


def update_review_metadata(file_dict, samples, current_index):
    reviewed = count_reviewed_by_current_reviewer(samples)
    file_dict["review_metadata"] = {
        "current_index": current_index,
        "total_samples": len(samples),
        "reviewed_samples_for_current_reviewer": reviewed,
        "review_finished_for_current_reviewer": reviewed == len(samples),
        "reviewer_id": REVIEWER_ID,
        "answer_key": ANSWER_KEY,
        "current_rendered_image": get_rendered_image_path(),
        "output_json": OUTPUT_JSON,
    }


def get_best_font(font_size):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                pass

    return ImageFont.load_default()


# ==========================================================
# FAST RENDERING (PIL)
# ==========================================================

def render_sample(sample, show_box=True):
    image_path = find_image_path(IMAGE_DIR, sample["image_name"])
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if show_box:
        xmin, ymin, xmax, ymax = sample["bbox"]
        class_name = sample["drawn_class_name"]

        box_w = max(1, xmax - xmin)
        box_h = max(1, ymax - ymin)
        min_dim = min(box_w, box_h)

        line_width = max(2, min(6, int(min_dim * 0.06)))
        font_size = max(12, min(28, int(min_dim * 0.35)))

        font = get_best_font(font_size)

        color = (255, 0, 0)
        text_color = (255, 255, 255)

        # Bounding box
        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline=color,
            width=line_width
        )

        # Textgröße exakt bestimmen
        text_bbox = draw.textbbox((0, 0), class_name, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        pad_x = max(4, font_size // 5)
        pad_y = max(2, font_size // 7)

        label_x1 = xmin
        label_y1 = ymin - text_h - 2 * pad_y - 2
        label_x2 = xmin + text_w + 2 * pad_x
        label_y2 = label_y1 + text_h + 2 * pad_y

        # Falls oberhalb kein Platz ist, Label in die Box setzen
        if label_y1 < 0:
            label_y1 = ymin
            label_y2 = label_y1 + text_h + 2 * pad_y

        # Falls rechts aus dem Bild läuft
        if label_x2 > img.width:
            shift = label_x2 - img.width
            label_x1 -= shift
            label_x2 -= shift
            label_x1 = max(0, label_x1)
            label_x2 = max(label_x1 + 1, label_x2)

        draw.rectangle(
            [label_x1, label_y1, label_x2, label_y2],
            fill=color
        )

        draw.text(
            (label_x1 + pad_x, label_y1 + pad_y - 1),
            class_name,
            fill=text_color,
            font=font
        )

    out_path = get_rendered_image_path()
    img.save(out_path, format="JPEG", quality=95, subsampling=0)

    print("\nOpen this image:")
    print(out_path)


# ==========================================================
# UI
# ==========================================================

def print_prompt(sample):
    answers = sample.get(ANSWER_KEY, {})
    current = answers.get(REVIEWER_ID)

    print("\n" + "-" * 60)
    #print(f"Image: {sample['image_name']}\n")
    print()
    print(f"{sample['drawn_class_name']} or Not?\n")
    #print(f"Reviewer: {REVIEWER_ID}")
    #print(f"Box ID: {sample['box_id']}")
    
    #print(f"Current answer: {None if current is None else int(current * 100)}%")
    print("^: 0% - 1: 10% - 2: 20% - 3: 30% - 4: 40% - 5: 50% - 6: 60% - 7: 70% - 8: 80% - 9: 90% - 0: 100%")
    print("\n" + "-" * 60)
    print()
    print("[V]iew toggle   [B]ack   [Q]uit")
    print()
    print("Selection: ", end="", flush=True)


def ask_for_score(sample, toggle_callback):
    print_prompt(sample)

    while True:
        key = getch.getch()

        if key.lower() == "q":
            print("Q")
            return "quit", None

        if key.lower() == "b":
            print("B")
            return "back", None

        if key.lower() == "v":
            print(" [toggle view]")
            toggle_callback()
            print_prompt(sample)
            continue

        if key in KEY_TO_SCORE:
            print(f"{key} -> {int(KEY_TO_SCORE[key] * 100)}%")
            return "score", KEY_TO_SCORE[key]


def show_sample(idx, total, sample):
    view_mode = True

    def toggle_view():
        nonlocal view_mode
        view_mode = not view_mode
        render_sample(sample, show_box=view_mode)

    print(f"\nSample {idx + 1}/{total}")
    print(f"Reviewer: {REVIEWER_ID}")

    render_sample(sample, True)

    return ask_for_score(sample, toggle_view)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            file = json.load(f)
        print(f"Resuming from existing file: {OUTPUT_JSON}")
    else:
        with open(INPUT_JSON, "r") as f:
            file = json.load(f)
        print(f"Starting from input file: {INPUT_JSON}")

    if "all_samples" not in file:
        raise KeyError("Input JSON must contain the key 'all_samples'.")

    samples = file["all_samples"]

    ensure_answer_key_exists(samples)

    current_index = get_resume_index(samples)
    total = len(samples)

    update_review_metadata(file, samples, current_index)
    save_state(file)

    print(f"Start index for {REVIEWER_ID}: {current_index}/{total}")

    while current_index < total:
        sample = samples[current_index]

        if REVIEWER_ID in sample[ANSWER_KEY]:
            current_index += 1
            continue

        action, value = show_sample(current_index, total, sample)

        if action == "quit":
            break

        if action == "back":
            if current_index > 0:
                current_index -= 1
                prev = samples[current_index]
                if REVIEWER_ID in prev.get(ANSWER_KEY, {}):
                    removed = prev[ANSWER_KEY].pop(REVIEWER_ID)
                    print(
                        f"Removed previous answer {int(removed * 100)}% "
                        f"from sample {prev['box_id']} for {REVIEWER_ID}"
                    )
                else:
                    print("Previous sample had no stored answer for this reviewer.")

                update_review_metadata(file, samples, current_index)
                save_state(file)
            else:
                print("Already at the beginning — cannot go back.")
            continue

        if action == "score":
            sample[ANSWER_KEY][REVIEWER_ID] = value
            current_index += 1

            update_review_metadata(file, samples, current_index)

            if current_index % 5 == 0:
                save_state(file)

    update_review_metadata(file, samples, current_index)
    save_state(file)

    print(f"\nSaved results to: {OUTPUT_JSON}")
    print(f"Rendered image path: {get_rendered_image_path()}")
    print(f"Finished for {REVIEWER_ID}.")