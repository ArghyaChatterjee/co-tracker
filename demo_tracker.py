#!/usr/bin/env python3
# CoTracker3 demo: mask-seeded tracking by filtering tracks at query frame,
# plus per-frame overlay (outline + translucent mask) on the original frames.

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import cv2
import torch
import torch.nn.functional as F

from cotracker.utils.visualizer import Visualizer, read_video_from_path

# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_video_tensor(path: str, device: str, downscale: float = 1.0):
    """
    Returns:
      video_t: float tensor [1, T, C, H, W], RGB in [0,255]
      frames_np: uint8 ndarray [T, H0, W0, C] RGB at ORIGINAL resolution (for overlays)
    """
    frames_np = read_video_from_path(path)  # (T,H,W,C), uint8 RGB
    if frames_np is None:
        raise RuntimeError(f"Failed to read video: {path}")

    video_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2)[None].float()  # [1,T,C,H,W] at original
    if abs(downscale - 1.0) > 1e-6:
        T, C, H, W = video_t.shape[1:]
        H2, W2 = int(round(H * downscale)), int(round(W * downscale))
        if H2 <= 0 or W2 <= 0:
            raise ValueError("--downscale must make H and W > 0. Try 1.0 or a small positive fraction.")
        video_t = F.interpolate(
            video_t.view(T, C, H, W), (H2, W2), mode="bilinear", align_corners=False
        ).view(1, T, C, H2, W2)
    return video_t.to(device), frames_np  # (processed tensor, original frames)

def load_binary_mask(mask_path: str, size_hw: tuple[int, int]) -> np.ndarray:
    """
    Loads mask image, converts to binary {0,1}, resizes to HxW, returns (H,W) uint8.
    """
    H, W = size_hw
    m = Image.open(mask_path).convert("L")
    if (m.size[1], m.size[0]) != (H, W):
        m = m.resize((W, H), Image.NEAREST)
    m = np.array(m)
    m = (m > 127).astype(np.uint8)
    return m  # (H,W) uint8 in {0,1}

def tracks_to_masks_filtered(pred_tracks, pred_visibility, H, W, vis_thr, keep_idx, min_pts=8):
    """
    pred_tracks:      [1, T, N, 2]
    pred_visibility:  [1, T, N, 1] or [1, T, N]
    keep_idx:         1-D integer indices of tracks to keep (those starting in your mask)
    Returns list of T masks at processed resolution (H, W), uint8 in {0,255}.
    """
    tracks = pred_tracks[0]  # (T, N, 2) torch
    vis = pred_visibility[0]  # (T, N) or (T, N, 1) torch
    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis[..., 0]  # (T, N)

    tracks = tracks[:, keep_idx, :]  # (T, N_kept, 2)
    vis = vis[:, keep_idx]          # (T, N_kept)

    masks = []
    for t in range(tracks.shape[0]):
        pts = tracks[t].detach().cpu().numpy()      # (N_kept, 2) -> (x, y)
        v   = vis[t].detach().cpu().numpy()         # (N_kept,)

        good = v > vis_thr
        pts_good = pts[good]
        if pts_good.shape[0] < min_pts:
            pts_good = pts  # relax

        m = np.zeros((H, W), np.uint8)
        if pts_good.shape[0] >= 3:
            hull = cv2.convexHull(pts_good.astype(np.float32).reshape(-1, 1, 2))
            cv2.fillConvexPoly(m, hull.astype(np.int32), 255)
        masks.append(m)
    return masks

def indices_inside_mask_at_frame(pred_tracks: torch.Tensor,
                                 t0: int,
                                 mask_hw: np.ndarray) -> np.ndarray:
    """
    Returns integer indices of tracks that start inside mask at frame t0.
    pred_tracks: [1,T,N,2] pixel coords (processed size)
    mask_hw: (H,W) uint8 in {0,1} (processed size)
    """
    xy0 = pred_tracks[0, t0].detach().cpu().numpy()  # (N,2)
    H, W = mask_hw.shape
    xi = np.clip(np.round(xy0[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(xy0[:, 1]).astype(int), 0, H - 1)
    inside = (mask_hw[yi, xi] > 0)
    return np.where(inside)[0]

def overlay_on_frame(frame_rgb: np.ndarray,
                     mask_uint8: np.ndarray,
                     color_bgr=(0, 255, 0),
                     alpha: float = 0.35,
                     draw_outline: bool = True,
                     thickness: int = 2) -> np.ndarray:
    """
    Returns a BGR frame with translucent overlay + outline drawn from mask.
    frame_rgb: (H, W, 3) uint8 RGB (original size)
    mask_uint8: (H, W) uint8 {0,255} (same original size)
    """
    # Convert to BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = frame_bgr.copy()

    # Fill
    if alpha > 0:
        colored = np.zeros_like(frame_bgr)
        colored[:, :] = color_bgr
        overlay = np.where(mask_uint8[..., None] > 0, colored, overlay)
        frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    # Outline
    if draw_outline:
        contours, _ = cv2.findContours((mask_uint8 > 0).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_bgr, contours, -1, color_bgr, thickness, lineType=cv2.LINE_AA)
    return frame_bgr

# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser("CoTracker3 demo (mask-seeded filtering + overlay)")
    p.add_argument("--video_path", required=True, help="left_seq.mp4")
    p.add_argument("--mask_path", required=True, help="binary mask for frame grid_query_frame (white=object)")
    p.add_argument("--out_dir", default="./cotracker_out", help="output dir")
    p.add_argument("--grid_size", type=int, default=12)
    p.add_argument("--grid_query_frame", type=int, default=0)
    p.add_argument("--backward_tracking", action="store_true", help="offline only")
    p.add_argument("--vis_thr", type=float, default=0.5)
    p.add_argument("--downscale", type=float, default=1.0)
    p.add_argument("--offline", action="store_true", help="use offline model")
    p.add_argument("--checkpoint", default=None, help="optional path to checkpoint")
    p.add_argument("--use_v2_model", action="store_true", help="use CoTracker2.x (not recommended)")
    p.add_argument("--save_vis_video", action="store_true")
    p.add_argument("--overlay_alpha", type=float, default=0.35, help="transparency of mask fill")
    p.add_argument("--no_fill", action="store_true", help="draw only outline (no translucent fill)")
    p.add_argument("--outline_thickness", type=int, default=2)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")

    out_dir = ensure_dir(args.out_dir)

    # Load video: processed tensor + original frames (RGB)
    video, frames_orig_rgb = load_video_tensor(args.video_path, device=device, downscale=args.downscale)
    T = video.shape[1]
    H_proc, W_proc = video.shape[-2], video.shape[-1]
    H_orig, W_orig = frames_orig_rgb.shape[1], frames_orig_rgb.shape[2]
    print(f"[info] video: T={T}, processed=({H_proc}x{W_proc}), original=({H_orig}x{W_orig}), device={device}, downscale={args.downscale}")

    # Load initial mask (resized to processed HxW)
    init_mask_proc = load_binary_mask(args.mask_path, (H_proc, W_proc))  # 0/1

    # Load model
    if args.checkpoint is not None:
        from cotracker.predictor import CoTrackerPredictor
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=True)
        else:
            window_len = 60 if args.offline else 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=False,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model_name = "cotracker3_offline" if args.offline else "cotracker3_online"
        model = torch.hub.load("facebookresearch/co-tracker", model_name)

    model = model.to(device).eval()

    # Prepare kwargs safely (no segm_mask for online; backward only for offline)
    call_kwargs = dict(grid_size=args.grid_size, grid_query_frame=args.grid_query_frame)
    if args.offline and args.backward_tracking:
        call_kwargs["backward_tracking"] = True

    # Run CoTracker
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, **call_kwargs)  # [1,T,N,2], [1,T,N,1] or [1,T,N]

    # Choose tracks that start inside the mask at the query frame (processed size)
    keep_idx = indices_inside_mask_at_frame(pred_tracks, args.grid_query_frame, init_mask_proc)

    # Build masks at processed size (0/255)
    masks_proc = tracks_to_masks_filtered(
        pred_tracks, pred_visibility, H_proc, W_proc, args.vis_thr, keep_idx
    )

    # Save raw masks (processed size) and make overlays on ORIGINAL frames
    masks_dir = ensure_dir(out_dir / "masks_proc")
    overlay_dir = ensure_dir(out_dir / "overlay_frames")

    # Optional: try to write a video
    writer = None
    try:
        import imageio.v3 as iio
        writer = iio.get_writer(str(out_dir / "overlay_video.mp4"), fps=30, macro_block_size=1)
    except Exception:
        writer = None

    for t in range(T):
        # Save processed mask
        m_proc = masks_proc[t]
        Image.fromarray(m_proc).save(masks_dir / f"mask_{t:04d}.png")

        # Upscale mask back to original size (nearest) for overlay
        m_orig = cv2.resize(m_proc, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

        # Compose overlay on original RGB frame
        frame_rgb = frames_orig_rgb[t]  # (H_orig,W_orig,3) RGB
        overlay_bgr = overlay_on_frame(
            frame_rgb,
            m_orig,
            color_bgr=(0, 255, 0),
            alpha=0.0 if args.no_fill else args.overlay_alpha,
            draw_outline=True,
            thickness=args.outline_thickness,
        )

        # Save overlay frame (PNG)
        cv2.imwrite(str(overlay_dir / f"frame_{t:04d}.png"), overlay_bgr)

        # Append to video if writer is available
        if writer is not None:
            # imageio expects RGB; convert back
            writer.append_data(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))

    if writer is not None:
        writer.close()

    # Optional track-visualization video (their built-in)
    if args.save_vis_video:
        vis = Visualizer(save_dir=str(out_dir), pad_value=120, linewidth=3)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if (args.offline and args.backward_tracking) else args.grid_query_frame,
        )

    print(f"[done] saved processed masks to {masks_dir}")
    print(f"[done] saved overlay frames to {overlay_dir}")
    if (out_dir / "overlay_video.mp4").exists():
        print(f"[done] overlay video: {out_dir / 'overlay_video.mp4'}")

if __name__ == "__main__":
    main()
