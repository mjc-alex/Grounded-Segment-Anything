import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, build_sam_hq, SamPredictor

# Recognize Anything Model & Tag2Text
from ram.models import ram, tag2text
from ram import inference_ram, inference_tag2text
import torchvision.transforms as TS


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

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def list_frames(scene_dir):
    frame_paths = [
        p for p in sorted(Path(scene_dir).iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return frame_paths


def parse_tags(tag_string):
    tags = [tag.strip().lower() for tag in tag_string.split(",")]
    return [tag for tag in tags if tag]


def sanitize_label(label):
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip().lower())
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


def build_scene_labels(
    frame_paths,
    label_source,
    topk_tags,
    min_freq_ratio,
    max_labels,
    device,
    ram_checkpoint,
    tag2text_checkpoint,
):
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])

    if label_source == "ram":
        tag_model = ram(pretrained=ram_checkpoint, image_size=384, vit="swin_l")
        tag_model.eval()
    else:
        delete_tag_index = list(range(3012, 3429))
        tag_model = tag2text(
            pretrained=tag2text_checkpoint,
            image_size=384,
            vit="swin_b",
            delete_tag_index=delete_tag_index,
        )
        tag_model.threshold = 0.64
        tag_model.eval()

    tag_model = tag_model.to(device)

    tag_counter = Counter()
    for frame_path in frame_paths:
        raw_image = Image.open(frame_path).convert("RGB").resize((384, 384))
        raw_tensor = transform(raw_image).unsqueeze(0).to(device)

        if label_source == "ram":
            res = inference_ram(raw_tensor, tag_model)
            tags = res[0].replace(" |", ",")
        else:
            res = inference_tag2text(raw_tensor, tag_model, specified_tags="None")
            tags = res[0].replace(" |", ",")

        tag_list = parse_tags(tags)
        if topk_tags is not None and topk_tags > 0:
            tag_list = tag_list[:topk_tags]
        tag_counter.update(set(tag_list))

    total_frames = len(frame_paths)
    min_count = max(1, int(total_frames * min_freq_ratio))
    filtered = [
        (tag, count)
        for tag, count in tag_counter.items()
        if count >= min_count
    ]
    filtered.sort(key=lambda item: (-item[1], item[0]))

    labels = [tag for tag, _ in filtered]
    if max_labels is not None and max_labels > 0:
        labels = labels[:max_labels]

    return labels


def validate_paths(scene_dir, output_dir, frame_paths):
    if not Path(scene_dir).exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    if not frame_paths:
        raise ValueError(f"No frames found in {scene_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(frame_paths)} frames in {scene_dir}")
    print(f"Output directory: {output_dir}")


def build_frame_output_dir(output_dir, frame_path):
    frame_folder = Path(output_dir) / frame_path.stem
    frame_folder.mkdir(parents=True, exist_ok=True)
    return frame_folder


def save_binary_mask(mask, output_path):
    mask_image = (mask.squeeze().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(mask_image).save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch pseudo-mask generator for scene frames")
    parser.add_argument("--scene_dir", type=str, required=True, help="path to scene frame folder")
    parser.add_argument("--output_dir", type=str, required=True, help="path to output root")
    parser.add_argument("--label_source", type=str, choices=["ram", "tag2text"], required=True)
    parser.add_argument("--sample_stride", type=int, default=1, help="frame stride for label sampling")
    parser.add_argument("--topk_tags", type=int, default=0, help="top-k tags per frame, 0=all")
    parser.add_argument("--min_freq_ratio", type=float, default=0.2, help="minimum frequency ratio")
    parser.add_argument("--max_labels", type=int, default=50, help="maximum labels in scene vocab")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--dry_run_n", type=int, default=0, help="limit to first N frames")
    parser.add_argument("--save_debug", action="store_true", help="save debug overlay images")
    parser.add_argument("--validate_paths", action="store_true", help="only validate paths")
    parser.add_argument("--config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounded_checkpoint", type=str, default="groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None)
    parser.add_argument("--use_sam_hq", action="store_true")
    parser.add_argument("--ram_checkpoint", type=str, default="ram_swin_large_14m.pth")
    parser.add_argument("--tag2text_checkpoint", type=str, default="tag2text_swin_14m.pth")
    args = parser.parse_args()

    all_frames = list_frames(args.scene_dir)
    if args.sample_stride > 1:
        sampled_frames = all_frames[:: args.sample_stride]
    else:
        sampled_frames = all_frames

    if args.dry_run_n and args.dry_run_n > 0:
        sampled_frames = sampled_frames[: args.dry_run_n]

    validate_paths(args.scene_dir, args.output_dir, all_frames)
    if args.validate_paths:
        raise SystemExit(0)

    labels = build_scene_labels(
        sampled_frames,
        args.label_source,
        args.topk_tags,
        args.min_freq_ratio,
        args.max_labels,
        args.device,
        args.ram_checkpoint,
        args.tag2text_checkpoint,
    )

    scene_labels_path = Path(args.output_dir) / "scene_labels.json"
    with open(scene_labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    frame_paths = sampled_frames if args.dry_run_n else all_frames
    label_filenames = {label: sanitize_label(label) + ".png" for label in labels}

    if not labels:
        for frame_path in frame_paths:
            build_frame_output_dir(args.output_dir, frame_path)
        print(f"Saved empty scene labels to {scene_labels_path}")
        raise SystemExit(0)

    model = load_grounding_dino(args.config, args.grounded_checkpoint, device=args.device)
    if args.use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=args.sam_hq_checkpoint).to(args.device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=args.sam_checkpoint).to(args.device))

    prompt = ", ".join(labels)
    for frame_path in frame_paths:
        image_pil, image = load_image(frame_path)
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, prompt, args.box_threshold, args.text_threshold, device=args.device
        )

        frame_dir = build_frame_output_dir(args.output_dir, frame_path)

        if boxes_filt.numel() == 0:
            for label, filename in label_filenames.items():
                output_path = frame_dir / filename
                save_binary_mask(
                    torch.zeros((1, image_pil.size[1], image_pil.size[0]), dtype=torch.bool),
                    output_path,
                )
            continue

        image_cv = np.array(image_pil)
        predictor.set_image(image_cv)

        size = image_pil.size
        height, width = size[1], size[0]
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

        label_to_masks = {label: [] for label in labels}
        for idx, phrase in enumerate(pred_phrases):
            label = phrase.split("(")[0].strip().lower()
            if label in label_to_masks:
                label_to_masks[label].append(masks[idx].bool())

        for label, filename in label_filenames.items():
            output_path = frame_dir / filename
            if not label_to_masks[label]:
                save_binary_mask(torch.zeros((1, height, width), dtype=torch.bool), output_path)
                continue
            merged = torch.zeros_like(label_to_masks[label][0])
            for mask in label_to_masks[label]:
                merged = torch.logical_or(merged, mask)
            save_binary_mask(merged.float(), output_path)

        if args.save_debug:
            debug_path = frame_dir / "debug.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write("\n".join(pred_phrases))

    print(f"Saved scene labels to {scene_labels_path}")
