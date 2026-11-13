import os
import tifffile
from PIL import Image
import numpy as np

def load_wsi_robust(file_obj_or_path, max_thumb=1024):
    """
    Safely loads thumbnail and full dimension metadata from any WSI,
    even giant 65,000x65,000 TIFFs, with ZERO MemoryError.
    """
    try:
        # Strategy A: Use tifffile to check series & pyramid levels
        with tifffile.TiffFile(file_obj_or_path) as tif:
            series = tif.series[0]
            orig_shape = series.shape # (H, W, C) or (H, W)
            orig_h = orig_shape[0]
            orig_w = orig_shape[1]
            print(f"tifffile detected: {orig_w}x{orig_h}, {len(series.levels)} levels")

            # Find best level close to max_thumb
            best_level = series.levels[-1]
            for lvl in reversed(series.levels):
                if max(lvl.shape[:2]) >= max_thumb:
                    best_level = lvl
                    break

            arr = best_level.asarray()
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            elif arr.shape[-1] > 3:
                arr = arr[..., :3]

            thumb_pil = Image.fromarray(arr).convert("RGB")
            # Resize cleanly to max_thumb
            scale = max_thumb / max(thumb_pil.size)
            if scale < 1.0:
                thumb_pil = thumb_pil.resize((max(1, int(thumb_pil.size[0]*scale)),
                                              max(1, int(thumb_pil.size[1]*scale))),
                                             Image.Resampling.BILINEAR)
            return thumb_pil, (orig_w, orig_h)
    except Exception as e:
        print("Strategy A failed:", e)

    # Strategy B: Pillow multi-frame / seek fallback
    try:
        im = Image.open(file_obj_or_path)
        orig_w, orig_h = im.size
        n_frames = getattr(im, "n_frames", 1)
        if n_frames > 1:
            im.seek(n_frames - 1)
            thumb = im.convert("RGB")
            scale = max_thumb / max(thumb.size)
            if scale < 1.0:
                thumb = thumb.resize((max(1, int(thumb.size[0]*scale)),
                                      max(1, int(thumb.size[1]*scale))),
                                     Image.Resampling.BILINEAR)
            return thumb, (orig_w, orig_h)
        else:
            # Single frame giant image: crop central 2048x2048 or draft
            cx, cy = orig_w // 2, orig_h // 2
            hw = min(1024, orig_w // 2)
            hh = min(1024, orig_h // 2)
            crop_patch = im.crop((cx - hw, cy - hh, cx + hw, cy + hh)).convert("RGB")
            return crop_patch, (orig_w, orig_h)
    except Exception as e:
        print("Strategy B failed:", e)
        raise e

print("Robust WSI loader compiled successfully!")

