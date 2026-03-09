"""CLI entry point for the Image & Video Quality Enhancement Pipeline."""

from __future__ import annotations

import argparse
import sys

import torch

from pipeline import DnCNN, FramePreprocessor, VideoPipeline, load_denoising_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ivqep",
        description=(
            "Image & Video Quality Enhancement Pipeline – "
            "apply contrast normalisation and CNN denoising to video frames, "
            "and optionally evaluate PSNR / SSIM quality metrics."
        ),
    )
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to write the enhanced output video (mp4v codec).  "
             "When omitted the video is processed but not saved.",
    )
    parser.add_argument(
        "--reference", default=None,
        help="Path to a clean reference video for quality metric evaluation.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a DnCNN checkpoint (state_dict).  "
             "When omitted CNN denoising is skipped.",
    )
    parser.add_argument(
        "--channels", type=int, default=3,
        help="Number of image channels for the CNN model (default: 3).",
    )
    parser.add_argument(
        "--num-features", type=int, default=64,
        help="Feature map count for the CNN model (default: 64).",
    )
    parser.add_argument(
        "--depth", type=int, default=17,
        help="Number of convolutional layers in the CNN (default: 17).",
    )
    parser.add_argument(
        "--clip-limit", type=float, default=2.0,
        help="CLAHE clip limit for contrast normalisation (default: 2.0).",
    )
    parser.add_argument(
        "--metrics", action="store_true",
        help="Compute per-frame PSNR and SSIM (requires --reference).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device string, e.g. 'cpu' or 'cuda:0'. "
             "Auto-selected when omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else None

    # --- Denoising model ---
    denoising_model: DnCNN | None = None
    if args.checkpoint:
        print(f"Loading DnCNN checkpoint from: {args.checkpoint}")
        denoising_model = load_denoising_model(
            checkpoint_path=args.checkpoint,
            channels=args.channels,
            num_features=args.num_features,
            depth=args.depth,
            device=device,
        )
    else:
        print("No checkpoint provided – CNN denoising will be skipped.")

    # --- Preprocessing ---
    preprocessor = FramePreprocessor(clip_limit=args.clip_limit)

    # --- Pipeline ---
    pipeline = VideoPipeline(
        preprocessor=preprocessor,
        denoising_model=denoising_model,
        device=device,
        compute_metrics=args.metrics,
    )

    print(f"Processing: {args.input}")
    if args.output:
        print(f"Output:     {args.output}")
    if args.metrics and args.reference:
        print(f"Reference:  {args.reference}")

    summary = pipeline.process_video(
        input_path=args.input,
        output_path=args.output,
        reference_path=args.reference,
        max_frames=args.max_frames,
    )

    print(f"\n{'─' * 40}")
    print(f"  Frames processed : {summary['total_frames']}")
    if summary["avg_psnr"] is not None:
        print(f"  Average PSNR     : {summary['avg_psnr']:.2f} dB")
    if summary["avg_ssim"] is not None:
        print(f"  Average SSIM     : {summary['avg_ssim']:.4f}")
    print(f"{'─' * 40}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
