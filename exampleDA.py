"""
Depth Anything V2 — Full Pipeline
==================================
Outputs (in a timestamped subfolder of SAVE_ROOT):
  - Depth_Map_Heatmap.png
  - Depth_Map_Contour.png
  - Depth_Map_Contour_3D.html   <- interactive, orbiteable in any browser
  - Object_Mesh.ply             (coloured, for visualisation)
  - Object_Solid.stl            (solid, 3-D printable; 1 mm flat base + depth relief)
  - crop_region.json            (saved crop path for this run)
  - crop_region_saved.json      (persistent across runs, in SAVE_ROOT)
  - crop_description.txt        (mathematical description for reports)

Crop tool (interactive matplotlib popup — mode chosen in terminal):
  - rectangle : click-drag
  - circle    : click centre, drag to set radius
  - spline    : left-click to add control points, right-click or Enter to close & fit
  Press 'r' to reset, Enter or close window to confirm.
"""

import cv2
import json
import os
import struct
import sys
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
from scipy.interpolate import splprep, splev

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_PATH   = r"Z:\Cas\Diepte_Herkenning\AIMODELS\Robotic-SPECTS\Prostate\prostate2.png"
DA_PATH      = r"Z:\Cas\Diepte_Herkenning\AIMODELS\Depth-Anything-V2"
SAVE_ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Example_Output")

ENCODER      = "vitl"    # "vits" | "vitb" | "vitl"
INDOOR_MODE  = False     # True = metric metres, False = relative depth
INVERT_DEPTH = False     # True -> closer = brighter in heatmap/PLY

DZ_SCALE     = 150       # Depth relief scale applied to normalised [0-1] depth
STL_BASE_MM  = 1.0       # Flat base thickness below the deepest point
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
class ReciprocalNorm(mcolors.Normalize):
    """1/z^alpha colour normalisation — close = bright, far = dark."""

    def __init__(self, vmin, vmax, alpha=1.0, clip=False):
        super().__init__(vmin, vmax, clip)
        self.alpha  = alpha
        self.r_vmin = 1.0 / (vmax ** alpha)
        self.r_vmax = 1.0 / (vmin ** alpha)

    def __call__(self, value, clip=None):
        value = np.array(value, dtype=float)
        value = np.where(value <= 0, np.nan, value)
        inv   = 1.0 / (value ** self.alpha)
        res   = (self.r_vmax - inv) / (self.r_vmax - self.r_vmin)
        return np.ma.masked_invalid(res)

    def inverse(self, value):
        inv = self.r_vmax - value * (self.r_vmax - self.r_vmin)
        return (1.0 / inv) ** (1.0 / self.alpha)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CROP TOOL
# ══════════════════════════════════════════════════════════════════════════════
class CropTool:
    """
    Interactive matplotlib crop selector.
    Mode is passed in as a string: 'rectangle', 'circle', or 'spline'.
    """

    def __init__(self, image_rgb: np.ndarray, mode: str = "rectangle"):
        self.image  = image_rgb
        self.H, self.W = image_rgb.shape[:2]
        self.mode   = mode
        self.result = None

        self._press       = None
        self._patch       = None
        self._spline_pts  = []
        self._spline_line = None
        self._pt_scatter  = None

        self._build_ui()

    def _build_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title(
            f"Crop [{self.mode}]  |  r=reset  |  Enter/close=confirm")
        self.ax.imshow(self.image)
        self.ax.set_title(
            f"Mode: {self.mode}  |  'r' = reset  |  Enter / close window = confirm")

        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_drag)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event",      self._on_key)

    def _clear_artists(self):
        for attr in ("_patch", "_spline_line", "_pt_scatter"):
            a = getattr(self, attr, None)
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
            setattr(self, attr, None)

    def _redraw(self):
        self.fig.canvas.draw_idle()

    def _reset(self):
        self._clear_artists()
        self._press      = None
        self._spline_pts = []
        self.result      = None
        self._redraw()

    def _on_press(self, event):
        if event.inaxes is not self.ax:
            return
        x, y = event.xdata, event.ydata
        if self.mode in ("rectangle", "circle"):
            if event.button == 1:
                self._press = (x, y)
        elif self.mode == "spline":
            if event.button == 1:
                self._spline_pts.append((x, y))
                self._draw_spline_preview()
            elif event.button == 3:
                self._close_spline()

    def _on_drag(self, event):
        if event.inaxes is not self.ax or self._press is None:
            return
        x0, y0 = self._press
        x1, y1 = event.xdata, event.ydata
        self._clear_artists()

        if self.mode == "rectangle":
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            self._patch = mpatches.Rectangle(
                (rx, ry), rw, rh,
                linewidth=2, edgecolor="cyan", facecolor="cyan", alpha=0.25)
            self.ax.add_patch(self._patch)

        elif self.mode == "circle":
            r = np.hypot(x1 - x0, y1 - y0)
            self._patch = mpatches.Circle(
                (x0, y0), r,
                linewidth=2, edgecolor="cyan", facecolor="cyan", alpha=0.25)
            self.ax.add_patch(self._patch)

        self._redraw()

    def _on_release(self, event):
        if self.mode == "spline" or self._press is None:
            return
        if event.inaxes is not self.ax:
            self._press = None
            return
        x0, y0 = self._press
        x1, y1 = event.xdata, event.ydata
        self._press = None

        if self.mode == "rectangle":
            self.result = {
                "type": "rectangle",
                "x": int(round(min(x0, x1))),
                "y": int(round(min(y0, y1))),
                "w": int(round(abs(x1 - x0))),
                "h": int(round(abs(y1 - y0))),
            }
        elif self.mode == "circle":
            self.result = {
                "type":   "circle",
                "cx":     float(x0),
                "cy":     float(y0),
                "radius": float(np.hypot(x1 - x0, y1 - y0)),
            }

    def _draw_spline_preview(self):
        if self._spline_line is not None:
            try: self._spline_line.remove()
            except Exception: pass
        if self._pt_scatter is not None:
            try: self._pt_scatter.remove()
            except Exception: pass

        pts = self._spline_pts
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        self._pt_scatter = self.ax.scatter(xs, ys, c="cyan", s=30, zorder=5)

        if len(pts) >= 3:
            arr = np.array(pts)
            try:
                tck, _ = splprep([arr[:, 0], arr[:, 1]], s=0, per=False)
                u_fine = np.linspace(0, 1, 400)
                sx, sy = splev(u_fine, tck)
                self._spline_line, = self.ax.plot(sx, sy, "c-", linewidth=2)
            except Exception:
                pass
        self._redraw()

    def _close_spline(self):
        pts = self._spline_pts
        if len(pts) < 3:
            print("  Need at least 3 points for a spline crop.")
            return
        arr = np.array(pts + [pts[0]])
        try:
            tck, _ = splprep([arr[:, 0], arr[:, 1]], s=0, per=True)
        except Exception as e:
            print(f"  Spline fit failed: {e}")
            return

        u_fine = np.linspace(0, 1, 800)
        sx, sy = splev(u_fine, tck)

        if self._spline_line is not None:
            try: self._spline_line.remove()
            except Exception: pass
        self._spline_line, = self.ax.plot(
            np.append(sx, sx[0]), np.append(sy, sy[0]), "c-", linewidth=2)
        self._redraw()

        self.result = {
            "type":           "spline",
            "control_points": pts,
            "curve_x":        sx.tolist(),
            "curve_y":        sy.tolist(),
        }
        self._spline_pts = []

    def _on_key(self, event):
        if event.key == "r":
            self._reset()
        elif event.key == "enter":
            if self.mode == "spline" and len(self._spline_pts) >= 3:
                self._close_spline()
            plt.close(self.fig)

    def run(self) -> dict | None:
        plt.show()
        return self.result


# ══════════════════════════════════════════════════════════════════════════════
#  CROP HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def build_mask(h: int, w: int, crop: dict) -> np.ndarray:
    """Return a boolean mask (H, W) — True = inside crop region."""
    mask = np.zeros((h, w), dtype=np.uint8)
    t    = crop["type"]

    if t == "rectangle":
        x, y, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
        x1, y1 = max(0, x),      max(0, y)
        x2, y2 = min(w, x + cw), min(h, y + ch)
        mask[y1:y2, x1:x2] = 1

    elif t == "circle":
        cx, cy, r = crop["cx"], crop["cy"], crop["radius"]
        yy, xx    = np.ogrid[:h, :w]
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 1

    elif t == "spline":
        curve_x  = np.array(crop["curve_x"])
        curve_y  = np.array(crop["curve_y"])
        pts_poly = np.column_stack([curve_x, curve_y]).astype(np.int32)
        cv2.fillPoly(mask, [pts_poly], 1)

    return mask.astype(bool)


def apply_crop(image: np.ndarray, depth: np.ndarray, crop: dict):
    """
    Crop image and depth to the bounding box of the crop region.
    Depth values are left exactly as they were in the full image — no scaling
    or zeroing anywhere.  The returned mask tells downstream code (STL builder)
    which pixels are inside the crop shape; everything else keeps its original
    depth so that visualisations show natural values at the crop boundary.
    Returns (cropped_image, cropped_depth, crop_mask_local, bbox).
    """
    H, W      = depth.shape
    mask_full = build_mask(H, W, crop)

    rows = np.any(mask_full, axis=1)
    cols = np.any(mask_full, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    cropped_depth = depth[r0:r1+1, c0:c1+1].copy()
    cropped_image = image[r0:r1+1, c0:c1+1].copy()
    cropped_mask  = mask_full[r0:r1+1, c0:c1+1]

    # Depth values are preserved everywhere (inside and outside the crop shape)
    # so that all visualisations show the natural depth at the crop boundary.
    # The STL builder uses crop_mask directly to set outside pixels to base_mm.

    return cropped_image, cropped_depth, cropped_mask, (r0, r1, c0, c1)


def describe_crop(crop: dict, image_shape: tuple) -> str:
    """Return a plain-ASCII mathematical description of the crop region."""
    H, W = image_shape[:2]
    t    = crop["type"]

    if t == "rectangle":
        x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
        return (
            f"Crop type     : Rectangle\n"
            f"Origin (x,y)  : ({x}, {y})  [pixels, top-left]\n"
            f"Size (w x h)  : {w} x {h} pixels\n"
            f"Right edge x  : {x + w}\n"
            f"Bottom edge y : {y + h}\n"
            f"Image size    : {W} x {H}\n"
            f"Crop area     : {w * h} px^2\n"
            f"Fraction      : {w * h / (W * H):.4f}"
        )
    elif t == "circle":
        cx, cy, r = crop["cx"], crop["cy"], crop["radius"]
        area = 3.14159265358979 * r * r
        return (
            f"Crop type     : Circle\n"
            f"Centre (cx,cy): ({cx:.1f}, {cy:.1f})  [pixels]\n"
            f"Radius        : {r:.1f} pixels\n"
            f"Area          : pi * r^2 = {area:.1f} px^2\n"
            f"Image size    : {W} x {H}\n"
            f"Fraction      : {area / (W * H):.4f}"
        )
    elif t == "spline":
        pts     = np.array(crop["control_points"])
        curve_x = np.array(crop["curve_x"])
        curve_y = np.array(crop["curve_y"])
        area    = 0.5 * abs(
            np.dot(curve_x, np.roll(curve_y, -1)) -
            np.dot(curve_y, np.roll(curve_x, -1))
        )
        dx    = np.diff(np.append(curve_x, curve_x[0]))
        dy    = np.diff(np.append(curve_y, curve_y[0]))
        perim = float(np.sum(np.hypot(dx, dy)))
        bbox_x = (float(curve_x.min()), float(curve_x.max()))
        bbox_y = (float(curve_y.min()), float(curve_y.max()))
        pts_str = "\n".join(f"  P{i:02d}: ({p[0]:.1f}, {p[1]:.1f})"
                            for i, p in enumerate(pts))
        return (
            f"Crop type          : Closed spline\n"
            f"Num control points : {len(pts)}\n"
            f"Control points     :\n{pts_str}\n"
            f"Bounding box x     : [{bbox_x[0]:.1f}, {bbox_x[1]:.1f}] px\n"
            f"Bounding box y     : [{bbox_y[0]:.1f}, {bbox_y[1]:.1f}] px\n"
            f"Enclosed area      : {area:.1f} px^2\n"
            f"Perimeter          : {perim:.1f} px\n"
            f"Image size         : {W} x {H}\n"
            f"Fraction           : {area / (W * H):.4f}"
        )
    return "Unknown crop type."


def save_crop(crop: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(crop, f, indent=2)
    print(f"  Crop saved        -> {path}")


def load_crop(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model():
    if DA_PATH not in sys.path:
        sys.path.insert(0, DA_PATH)

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")

    if INDOOR_MODE:
        from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        cfg  = {**model_configs[ENCODER], "max_depth": 10}
        ckpt = os.path.join(DA_PATH,
               f"checkpoints/depth_anything_v2_metric_hypersim_{ENCODER}.pth")
    else:
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg  = model_configs[ENCODER]
        ckpt = os.path.join(DA_PATH,
               f"checkpoints/depth_anything_v2_{ENCODER}.pth")

    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def renorm_depth(depth: np.ndarray, invert: bool = True) -> np.ndarray:
    if invert:
        depth = 1.0 / (depth + 1e-8)
    depth = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    if not invert:
        depth = 1.0 - depth
    return depth


# ══════════════════════════════════════════════════════════════════════════════
#  PLY MESH  (coloured, for visualisation)
# ══════════════════════════════════════════════════════════════════════════════
def build_ply_mesh(depth_map: np.ndarray,
                   rgb_image: np.ndarray | None = None,
                   cmap_name: str = "gray",
                   dz: float = 1.0) -> o3d.geometry.TriangleMesh:
    H, W     = depth_map.shape
    scaled   = depth_map * dz

    jj, ii   = np.meshgrid(np.arange(W), np.arange(H))
    vertices = np.column_stack([jj.ravel().astype(float),
                                ii.ravel().astype(float),
                                scaled.ravel().astype(float)])

    if rgb_image is not None:
        colours = rgb_image.reshape(-1, 3) / 255.0
    else:
        cmap    = plt.get_cmap(cmap_name)
        inv     = 1.0 / (depth_map + 1e-8)
        norm    = ReciprocalNorm(vmin=inv.min(), vmax=inv.max(), alpha=1.5)
        colours = np.array([cmap(float(v))[:3] for v in norm(inv).ravel()])

    idx  = np.arange(H * W).reshape(H, W)
    v0   = idx[:-1, :-1].ravel();  v1 = idx[:-1, 1:].ravel()
    v2   = idx[1:,  :-1].ravel();  v3 = idx[1:,  1:].ravel()
    tris = np.concatenate([np.column_stack([v0, v1, v2]),
                           np.column_stack([v1, v3, v2])], axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(vertices)
    mesh.triangles     = o3d.utility.Vector3iVector(tris)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colours)
    mesh.compute_vertex_normals()
    return mesh


# ══════════════════════════════════════════════════════════════════════════════
#  STL SOLID  (3-D printable, watertight)
# ══════════════════════════════════════════════════════════════════════════════
def depth_to_stl_z(depth_map: np.ndarray, dz: float, base_mm: float,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """
    Convert a normalised depth map to z-heights for the STL top surface.

    The depth map from renorm_depth() encodes larger values for closer pixels.
    For a convex object (protrudes toward camera), closer pixels should be
    HIGHER in z, so z is mapped directly: z = depth_map * dz.

    Steps:
      1. Scale: z = depth_map * dz  -> close pixels (high depth) get high z.
      2. Shift so the minimum z over the in-mask region = base_mm
         (ensures no zero-thickness print and correct floor level).
      3. Outside mask pixels are set to base_mm (flat floor, does not protrude).
    """
    z = depth_map * dz

    # Compute floor shift only over the region that matters (inside mask).
    if mask is not None:
        z_min = z[mask].min() if mask.any() else 0.0
    else:
        z_min = z.min()

    z = z - z_min + base_mm

    if mask is not None:
        z[~mask] = base_mm       # outside crop -> flat floor, not a peak

    return z


def build_solid_stl(depth_map: np.ndarray,
                    dz: float,
                    base_mm: float,
                    mask: np.ndarray | None = None) -> list:
    """Build a watertight solid. Top surface = depth relief, bottom = flat at z=0."""
    H, W      = depth_map.shape
    z_surface = depth_to_stl_z(depth_map, dz, base_mm, mask)

    triangles = []

    def tri(a, b, c):
        ab = b - a;  ac = c - a
        n  = np.cross(ab, ac)
        ln = np.linalg.norm(n)
        n  = (n / ln) if ln > 0 else n
        triangles.append((n, a, b, c))

    # Top surface
    for i in range(H - 1):
        for j in range(W - 1):
            p00 = np.array([j,   i,   z_surface[i,   j  ]], dtype=float)
            p10 = np.array([j+1, i,   z_surface[i,   j+1]], dtype=float)
            p01 = np.array([j,   i+1, z_surface[i+1, j  ]], dtype=float)
            p11 = np.array([j+1, i+1, z_surface[i+1, j+1]], dtype=float)
            tri(p00, p10, p01)
            tri(p10, p11, p01)

    # Bottom face (z=0, reversed winding -> normal points down)
    for i in range(H - 1):
        for j in range(W - 1):
            b00 = np.array([j,   i,   0.0], dtype=float)
            b10 = np.array([j+1, i,   0.0], dtype=float)
            b01 = np.array([j,   i+1, 0.0], dtype=float)
            b11 = np.array([j+1, i+1, 0.0], dtype=float)
            tri(b00, b01, b10)
            tri(b10, b01, b11)

    # Side walls
    for i in range(H - 1):   # left (j=0)
        t  = np.array([0, i,   z_surface[i,   0]], dtype=float)
        b  = np.array([0, i,   0.0],               dtype=float)
        tn = np.array([0, i+1, z_surface[i+1, 0]], dtype=float)
        bn = np.array([0, i+1, 0.0],               dtype=float)
        tri(t, b, tn);  tri(b, bn, tn)

    for i in range(H - 1):   # right (j=W-1)
        t  = np.array([W-1, i,   z_surface[i,   W-1]], dtype=float)
        b  = np.array([W-1, i,   0.0],                 dtype=float)
        tn = np.array([W-1, i+1, z_surface[i+1, W-1]], dtype=float)
        bn = np.array([W-1, i+1, 0.0],                 dtype=float)
        tri(t, tn, b);  tri(b, tn, bn)

    for j in range(W - 1):   # top edge (i=0)
        t  = np.array([j,   0, z_surface[0, j  ]], dtype=float)
        b  = np.array([j,   0, 0.0],               dtype=float)
        tn = np.array([j+1, 0, z_surface[0, j+1]], dtype=float)
        bn = np.array([j+1, 0, 0.0],               dtype=float)
        tri(t, tn, b);  tri(b, tn, bn)

    for j in range(W - 1):   # bottom edge (i=H-1)
        t  = np.array([j,   H-1, z_surface[H-1, j  ]], dtype=float)
        b  = np.array([j,   H-1, 0.0],                 dtype=float)
        tn = np.array([j+1, H-1, z_surface[H-1, j+1]], dtype=float)
        bn = np.array([j+1, H-1, 0.0],                 dtype=float)
        tri(t, b, tn);  tri(b, bn, tn)

    return triangles


def write_stl(triangles: list, path: str):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for (n, a, b, c) in triangles:
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *a))
            f.write(struct.pack("<3f", *b))
            f.write(struct.pack("<3f", *c))
            f.write(struct.pack("<H", 0))
    print(f"  Saved STL         -> {path}  ({len(triangles):,} triangles)")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def save_heatmap(depth: np.ndarray, path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(depth, cmap="gray_r", vmin=depth.min(), vmax=depth.max())
    divider = make_axes_locatable(ax)
    fig.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.05)
                 ).set_label("~r (no units)")
    ax.set_title("Depth Map - Heatmap")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap     -> {path}")


def save_contour(depth: np.ndarray, path: str, levels: int = 150):
    H, W  = depth.shape
    x, y  = np.meshgrid(np.arange(W), np.arange(H))
    dpos  = depth[depth > 0]
    if len(dpos) == 0:
        print("  Contour skipped (no positive depth values).")
        return
    lvls = np.logspace(np.log10(dpos.min() + 1e-6),
                       np.log10(depth.max()), levels)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    c = ax.contour(x, y, depth, colors="black", levels=lvls, linewidths=0.8)
    plt.clabel(c, inline=True, fontsize=7, levels=c.levels[::2])
    ax.set_title("Depth Map - Contour")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved contour     -> {path}")


def save_contour_3d_html(depth: np.ndarray, path: str, step: int = 4):
    """
    Export an interactive, orbiteable 3-D surface plot with contour lines as a
    self-contained HTML file using Plotly.  Contour lines are drawn directly on
    the surface so the plot acts as both a depth surface and a contour map.
    'step' downsamples the depth map so the file stays manageable
    (step=4 -> uses every 4th pixel in x and y).
    """
    d = depth[::step, ::step]
    H, W = d.shape
    x = np.arange(W) * step
    y = np.arange(H) * step

    fig = go.Figure(data=[go.Surface(
        z=d,
        x=x,
        y=y,
        colorscale="Viridis",
        colorbar=dict(title="Depth"),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="white",
                project_z=False,
                width=2,
            )
        ),
    )])

    fig.update_layout(
        title="Depth Map - Interactive 3D Contour Surface (orbit with mouse)",
        scene=dict(
            xaxis_title="X (px)",
            yaxis_title="Y (px)",
            zaxis_title="Depth",
            aspectmode="manual",
            aspectratio=dict(x=W / max(H, W), y=H / max(H, W), z=0.4),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print(f"  Saved 3D HTML     -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(SAVE_ROOT, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    crop_file        = os.path.join(save_folder, "crop_region.json")
    global_crop_file = os.path.join(SAVE_ROOT,   "crop_region_saved.json")

    print(f"\nOutput folder: {save_folder}\n")

    # ── Load image ────────────────────────────────────────────────────────────
    bgr_frame = cv2.imread(IMAGE_PATH)
    if bgr_frame is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {rgb_frame.shape[1]} x {rgb_frame.shape[0]} px")

    # ── Crop selection ────────────────────────────────────────────────────────
    print("\nCrop options:")
    print("  1 - Rectangle")
    print("  2 - Circle")
    print("  3 - Spline (click points, right-click or Enter to close)")
    print("  4 - Load previously saved crop")
    print("  5 - No crop (use full image)")
    choice = input("Select [1-5]: ").strip()

    crop = None

    if choice == "5":
        print("  Using full image.")

    elif choice == "4":
        crop = load_crop(global_crop_file)
        if crop is None:
            print(f"  No saved crop found. Using full image.")
        else:
            print(f"  Loaded saved crop: {crop['type']}")

    else:
        mode_map = {"1": "rectangle", "2": "circle", "3": "spline"}
        mode     = mode_map.get(choice, "rectangle")
        print(f"  Opening crop tool in [{mode}] mode ...")
        tool = CropTool(rgb_frame, mode=mode)
        crop = tool.run()

        if crop is None:
            print("  No crop drawn — using full image.")
        else:
            save_crop(crop, crop_file)
            save_crop(crop, global_crop_file)

    # ── Describe & log crop ───────────────────────────────────────────────────
    if crop:
        desc = describe_crop(crop, rgb_frame.shape)
        print("\n-- Crop Description " + "-" * 40)
        print(desc)
        print("-" * 59)
        desc_path = os.path.join(save_folder, "crop_description.txt")
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write(desc + "\n")
        print(f"  Saved description -> {desc_path}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nLoading model ...")
    model = load_model()
    print("Running inference ...")
    raw_depth = model.infer_image(bgr_frame)
    depth     = renorm_depth(raw_depth, invert=INVERT_DEPTH)
    # depth is now normalised to [0, 1]; closer pixels have larger values.

    # ── Apply crop ────────────────────────────────────────────────────────────
    if crop:
        rgb_work, depth_work, crop_mask, bbox = apply_crop(rgb_frame, depth, crop)
        print(f"  Cropped to rows {bbox[0]}-{bbox[1]}, cols {bbox[2]}-{bbox[3]}")
    else:
        rgb_work   = rgb_frame
        depth_work = depth
        crop_mask  = None

    # ── PLY mesh (visualisation) ──────────────────────────────────────────────
    # PLY inverts depth (max - depth) so far pixels (low depth) become high z,
    # giving the correct 3-D shape when orbited in a PLY viewer from any angle.
    print("\nBuilding PLY mesh ...")
    ply_depth = np.flipud(np.max(depth_work) - depth_work)
    ply_rgb   = np.flipud(rgb_work)
    ply_mesh  = build_ply_mesh(ply_depth, rgb_image=ply_rgb, dz=DZ_SCALE)
    ply_path  = os.path.join(save_folder, "Object_Mesh.ply")
    o3d.io.write_triangle_mesh(ply_path, ply_mesh)
    print(f"  Saved PLY mesh    -> {ply_path}")

    # ── STL solid ─────────────────────────────────────────────────────────────
    # depth_work: larger values = closer. depth_to_stl_z() maps high depth ->
    # high z so the close/near region protrudes upward (convex relief).
    print("Building STL solid ...")
    stl_tris = build_solid_stl(depth_work, dz=DZ_SCALE,
                               base_mm=STL_BASE_MM, mask=crop_mask)
    stl_path = os.path.join(save_folder, "Object_Solid.stl")
    write_stl(stl_tris, stl_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nSaving plots ...")
    save_heatmap(depth_work,
                 os.path.join(save_folder, "Depth_Map_Heatmap.png"))
    save_contour(depth_work,
                 os.path.join(save_folder, "Depth_Map_Contour.png"))
    save_contour_3d_html(depth_work,
                         os.path.join(save_folder, "Depth_Map_Contour_3D.html"))

    print(f"\nAll done! Files saved to:\n  {save_folder}")


if __name__ == "__main__":
    run()
  
