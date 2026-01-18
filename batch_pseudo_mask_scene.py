import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

# segment anything
from segment_anything import build_sam, build_sam_hq, SamPredictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_grounding_dino(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    scores = logits_filt.max(dim=1)[0]
    return boxes_filt, scores


def list_frames(scene_dir):
    return [
        p for p in sorted(Path(scene_dir).iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def list_jsons(json_dir):
    return [p for p in sorted(Path(json_dir).iterdir()) if p.suffix.lower() == ".json"]


def load_scene_labels(json_paths):
    labels = OrderedDict()
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for obj in payload.get("object", []):
            category = obj.get("category", "").strip()
            segmentation = obj.get("segmentation", "").strip()
            sentences = obj.get("sentence", [])
            if not category or not segmentation:
                continue
            filename = Path(segmentation).name
            if category in labels and labels[category]["filename"] != filename:
                raise ValueError(
                    f"Category '{category}' has inconsistent segmentation names: "
                    f"{labels[category]['filename']} vs {filename}"
                )
            if category not in labels:
                labels[category] = {
                    "category": category,
                    "filename": filename,
                    "sentence": sentences[0] if sentences else "",
                }
    return list(labels.values())


def validate_paths(scene_dir, json_dir, output_dir, frame_paths, json_paths):
    scene_path = Path(scene_dir)
    json_path = Path(json_dir)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not frame_paths:
        raise ValueError(f"No frames found in {scene_dir}")
    if not json_paths:
        raise ValueError(f"No JSON files found in {json_dir}")

    frame_stems = {p.stem for p in frame_paths}
    json_stems = {p.stem for p in json_paths}
    missing_json = sorted(frame_stems - json_stems)
    if missing_json:
        raise ValueError(f"Missing JSON files for frames: {missing_json[:5]}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(frame_paths)} frames in {scene_dir}")
    print(f"Found {len(json_paths)} JSON files in {json_dir}")
    print(f"Output directory: {output_dir}")


def build_frame_output_dir(output_dir, frame_path):
    frame_folder = Path(output_dir) / frame_path.stem
    frame_folder.mkdir(parents=True, exist_ok=True)
    return frame_folder


def save_binary_mask(mask, output_path):
    mask_image = (mask.squeeze().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(mask_image).save(output_path)


def load_frame_objects(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    objects = {}
    for obj in payload.get("object", []):
        category = obj.get("category", "").strip()
        sentences = obj.get("sentence", [])
        if not category:
            continue
        objects[category] = {
            "sentence": sentences[0] if sentences else "",
        }
    return objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch pseudo-mask generator for scene frames")
    parser.add_argument("--scene_dir", type=str, required=True, help="path to scene frame folder")
    parser.add_argument("--json_dir", type=str, required=True, help="path to frame json folder")
    parser.add_argument("--output_dir", type=str, required=True, help="path to output root")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--dry_run_n", type=int, default=0, help="limit to first N frames")
    parser.add_argument("--save_debug", action="store_true", help="save debug phrases")
    parser.add_argument("--validate_paths", action="store_true", help="only validate paths")
    parser.add_argument(
        "--config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument("--grounded_checkpoint", type=str, default="groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None)
    parser.add_argument("--use_sam_hq", action="store_true")
    args = parser.parse_args()

    all_frames = list_frames(args.scene_dir)
    all_jsons = list_jsons(args.json_dir)
    validate_paths(args.scene_dir, args.json_dir, args.output_dir, all_frames, all_jsons)
    if args.validate_paths:
        raise SystemExit(0)

    frame_paths = all_frames
    if args.dry_run_n and args.dry_run_n > 0:
        frame_paths = frame_paths[: args.dry_run_n]

    json_by_stem = {p.stem: p for p in all_jsons}
    scene_labels = load_scene_labels([json_by_stem[p.stem] for p in frame_paths])

    scene_labels_path = Path(args.output_dir) / "scene_labels.json"
    with open(scene_labels_path, "w", encoding="utf-8") as f:
        json.dump(scene_labels, f, ensure_ascii=False, indent=2)

    label_filenames = {label["category"]: label["filename"] for label in scene_labels}

    model = load_grounding_dino(args.config, args.grounded_checkpoint, device=args.device)
    if args.use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=args.sam_hq_checkpoint).to(args.device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=args.sam_checkpoint).to(args.device))

    for frame_path in frame_paths:
        image_pil, image = load_image(frame_path)
        height, width = image_pil.size[1], image_pil.size[0]
        frame_dir = build_frame_output_dir(args.output_dir, frame_path)

        frame_objects = load_frame_objects(json_by_stem[frame_path.stem])

        image_cv = np.array(image_pil)
        predictor.set_image(image_cv)

        debug_lines = []
        for label in scene_labels:
            category = label["category"]
            output_name = label_filenames[category]
            output_path = frame_dir / output_name

            if category not in frame_objects:
                save_binary_mask(torch.zeros((1, height, width), dtype=torch.bool), output_path)
                continue

            sentence = frame_objects[category]["sentence"].strip()
            if not sentence:
                save_binary_mask(torch.zeros((1, height, width), dtype=torch.bool), output_path)
                continue

            boxes_filt, scores = get_grounding_output(
                model,
                image,
                sentence,
                args.box_threshold,
                args.text_threshold,
                device=args.device,
            )

            if boxes_filt.numel() == 0:
                save_binary_mask(torch.zeros((1, height, width), dtype=torch.bool), output_path)
                continue

            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([width, height, width, height])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(args.device)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(args.device),
                multimask_output=False,
            )

            merged = torch.zeros_like(masks[0]).bool()
            for mask in masks:
                merged = torch.logical_or(merged, mask.bool())

            save_binary_mask(merged.float(), output_path)
            debug_lines.append(f"{category}: {sentence} ({scores.max().item():.3f})")

        if args.save_debug:
            debug_path = frame_dir / "debug.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write("\n".join(debug_lines))

    print(f"Saved scene labels to {scene_labels_path}")
