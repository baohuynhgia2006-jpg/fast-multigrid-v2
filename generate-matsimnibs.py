"""
Generate SimNIBS 4x4 matsimnibs.txt files directly from final_tissue_MNI.nii.gz
for use with the C++ multigrid TMS solver (main_3.cpp).

Dependencies:
    pip install numpy nibabel scipy scikit-image
    (Falls back to scipy binary_erosion if scikit-image is not installed.)
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter, map_coordinates

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def load_scalp_surface(nii_path: str):
    """Load NIfTI, build head mask matching charm_label_to_sigma (1..10), and extract scalp vertices (mm)."""
    img = nib.load(nii_path)
    affine = img.affine.astype(np.float64)
    inv_affine = np.linalg.inv(affine)

    # Check voxel isotropy (C++ FEM stencil assumes cubical voxels h = dx)
    zooms = img.header.get_zooms()[:3]
    print(f"Loaded {nii_path}: shape={img.shape[:3]}, spacing={tuple(round(float(z), 4) for z in zooms)} mm")
    if abs(zooms[0] - zooms[1]) > 1e-2 or abs(zooms[0] - zooms[2]) > 1e-2:
        print("Warning: Voxels are non-isotropic; main_3.cpp assumes isotropic spacing h = dx.")

    zooms = img.header.get_zooms()[:3]
    print(f"Loaded {nii_path}: shape={img.shape}, spacing={tuple(round(float(z), 4) for z in zooms)} mm")
    if abs(zooms[0] - zooms[1]) > 1e-2 or abs(zooms[0] - zooms[2]) > 1e-2:
        print("Warning: Voxels are non-isotropic; main_3.cpp assumes isotropic spacing h = dx.")

    # Strip trailing singleton dimensions (e.g. (182, 218, 182, 1) -> (182, 218, 182))
    raw = img.get_fdata()
    while raw.ndim > 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3D tissue label volume, got shape {raw.shape}.")

    # Match C++ charm_label_to_sigma: labels 1..10 are conducting tissue, others are air (0)
    labels = np.rint(raw).astype(np.int32)
    head_mask = binary_fill_holes((labels >= 1) & (labels <= 10))
    if not np.any(head_mask):
        raise ValueError("No conducting tissue labels (1..10) found in NIfTI volume.")

    if HAS_SKIMAGE:
        smoothed = gaussian_filter(head_mask.astype(np.float32), sigma=1.0)
        verts_vox, _, _, _ = marching_cubes(smoothed, level=0.5)
    else:
        print("Note: scikit-image not found; using binary_erosion fallback for scalp surface.")
        surface_mask = head_mask & ~binary_erosion(head_mask)
        verts_vox = np.argwhere(surface_mask).astype(np.float64)

    verts_mm = nib.affines.apply_affine(affine, verts_vox)
    return verts_mm, head_mask.astype(np.float32), inv_affine


def scalp_point_and_normal(
    ref_mm,
    verts_mm: np.ndarray,
    head_mask: np.ndarray,
    inv_affine: np.ndarray,
    patch_radius: float = 10.0,
):
    """Find nearest scalp point to ref_mm and its smoothed outward unit normal."""
    ref_mm = np.asarray(ref_mm, dtype=np.float64)
    dists = np.linalg.norm(verts_mm - ref_mm, axis=1)
    p_scalp = verts_mm[np.argmin(dists)]

    # Fit local tangent plane via SVD over scalp patch
    patch_dists = np.linalg.norm(verts_mm - p_scalp, axis=1)
    patch = verts_mm[patch_dists <= patch_radius]
    if len(patch) < 10:
        patch = verts_mm[np.argsort(patch_dists)[:50]]

    _, _, vh = np.linalg.svd(patch - patch.mean(axis=0), full_matrices=False)
    n_out = vh[-1]
    n_out /= np.linalg.norm(n_out)

    # Orient normal outward (toward air = 0, away from head tissue = 1)
    def sample_mask(pt_mm):
        pt_vox = nib.affines.apply_affine(inv_affine, pt_mm)
        return float(map_coordinates(head_mask, pt_vox[:, None], order=1, mode="constant", cval=0.0)[0])

    pos_score = sample_mask(p_scalp + 3.0 * n_out) + sample_mask(p_scalp + 6.0 * n_out)
    neg_score = sample_mask(p_scalp - 3.0 * n_out) + sample_mask(p_scalp - 6.0 * n_out)
    if pos_score > neg_score:
        n_out = -n_out

    return p_scalp, n_out


def build_matsimnibs(coil_center, n_out, handle_vec=(0.0, -1.0, 0.0)) -> np.ndarray:
    """
    Build right-handed 4x4 matsimnibs matrix matching SimNIBS & main_3.cpp:
      - Column 2 (+z): inward toward the head (-n_out)
      - Column 1 (+y): handle_vec projected onto the scalp tangent plane
      - Column 0 (+x): py x pz (right-handed, det(R) = +1)
      - Column 3 (t):  coil center in MNI RAS+ millimeters
    """
    pz = -np.asarray(n_out, dtype=np.float64)
    pz /= np.linalg.norm(pz)

    v = np.asarray(handle_vec, dtype=np.float64)
    py = v - np.dot(v, pz) * pz
    py_norm = np.linalg.norm(py)
    if py_norm < 1e-6:
        raise ValueError("handle_vec is parallel to the local scalp normal; cannot define coil +y axis.")
    py /= py_norm

    px = np.cross(py, pz)
    px /= np.linalg.norm(px)

    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = px
    mat[:3, 1] = py
    mat[:3, 2] = pz
    mat[:3, 3] = np.asarray(coil_center, dtype=np.float64)
    return mat


def verify_matsimnibs(
    mat: np.ndarray,
    head_mask: np.ndarray,
    inv_affine: np.ndarray,
    wing_radius_mm: float = 35.0,
    label: str = "",
):
    """Sanity-check orthonormality, inward +z orientation, air clearance, and wing clipping."""
    R = mat[:3, :3]
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    det_R = float(np.linalg.det(R))
    if ortho_err > 1e-5 or not np.isclose(det_R, 1.0, atol=1e-5):
        raise AssertionError(f"[{label}] Invalid rotation matrix: ortho_err={ortho_err:.2e}, det={det_R:.6f}")

    def mask_at_coil_local(pt_local):
        pt_world = (mat @ np.array([pt_local[0], pt_local[1], pt_local[2], 1.0]))[:3]
        vox = np.rint(nib.affines.apply_affine(inv_affine, pt_world)).astype(int)
        if np.any(vox < 0) or np.any(vox >= head_mask.shape):
            return 0.0
        return float(head_mask[tuple(vox)])

    if mask_at_coil_local((0.0, 0.0, 0.0)) > 0.0:
        raise AssertionError(f"[{label}] Coil center is inside head tissue!")
    if mask_at_coil_local((0.0, 0.0, 15.0)) == 0.0:
        print(f"Warning [{label}]: +15 mm along coil +z did not hit head tissue (check offset or normal).")

    clipped = []
    for deg in range(0, 360, 45):
        rad = np.deg2rad(deg)
        pt = (wing_radius_mm * np.cos(rad), wing_radius_mm * np.sin(rad), 0.0)
        if mask_at_coil_local(pt) > 0.0:
            clipped.append(deg)
    if clipped:
        print(f"Warning [{label}]: coil plane (r={wing_radius_mm} mm) clips scalp at angles {clipped} deg.")
    else:
        print(f"Verification passed [{label}]: right-handed (det={det_R:.6f}), inward +z, clear of scalp.")


def main():
    parser = argparse.ArgumentParser(description="Compute matsimnibs.txt from final_tissue_MNI.nii.gz.")
    parser.add_argument("nifti", nargs="?", default="final_tissue_MNI.nii.gz",
                        help="Path to final_tissue_MNI.nii(.gz)")
    parser.add_argument("--target", type=float, nargs=3, default=[-38.0, -22.0, 56.0],
                        help="Cortical MNI target (mm), default: -38 -22 56")
    parser.add_argument("--offset", type=float, default=5.0,
                        help="Scalp-to-coil offset (mm), default: 5.0")
    parser.add_argument("--fixed-center", type=float, nargs=3, default=[-54.62, -23.31, 79.5],
                        help="Fixed coil center in MNI (mm), default: -54.62 -23.31 79.5")
    parser.add_argument("--handle", type=float, nargs=3, default=[0.0, -1.0, 0.0],
                        help="Desired coil +y direction vector, default: 0 -1 0")
    parser.add_argument("--patch-radius", type=float, default=10.0,
                        help="Scalp patch radius (mm) for normal smoothing, default: 10.0")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Output directory for matsimnibs text files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts_mm, head_mask, inv_affine = load_scalp_surface(args.nifti)

    # 1. Coil placement from MNI cortical target + scalp offset
    p_scalp_t, n_out_t = scalp_point_and_normal(
        args.target, verts_mm, head_mask, inv_affine, patch_radius=args.patch_radius
    )
    coil_center_t = p_scalp_t + args.offset * n_out_t
    mat_target = build_matsimnibs(coil_center_t, n_out_t, handle_vec=args.handle)

    out_target = out_dir / "matsimnibs_from_target.txt"
    np.savetxt(out_target, mat_target, fmt="%.6f")

    print("\n--- 1. From MNI Target + Scalp Offset ---")
    print(f"Target MNI (mm):       {np.round(args.target, 3)}")
    print(f"Nearest scalp (mm):    {np.round(p_scalp_t, 3)}")
    print(f"Outward normal:        {np.round(n_out_t, 6)}")
    print(f"Coil center (+{args.offset} mm): {np.round(coil_center_t, 3)}")
    print("matsimnibs:\n", np.array2string(mat_target, precision=6, suppress_small=True))
    verify_matsimnibs(mat_target, head_mask, inv_affine, label="target")
    print(f"Saved: {out_target}")

    # 2. Coil placement at the fixed coil center
    fixed_center = np.asarray(args.fixed_center, dtype=np.float64)
    p_scalp_f, n_out_f = scalp_point_and_normal(
        fixed_center, verts_mm, head_mask, inv_affine, patch_radius=args.patch_radius
    )
    actual_offset = float(np.linalg.norm(fixed_center - p_scalp_f))
    mat_fixed = build_matsimnibs(fixed_center, n_out_f, handle_vec=args.handle)

    out_fixed = out_dir / "matsimnibs_fixed.txt"
    np.savetxt(out_fixed, mat_fixed, fmt="%.6f")

    print("\n--- 2. From Fixed Coil Center ---")
    print(f"Fixed center (mm):     {np.round(fixed_center, 3)}")
    print(f"Nearest scalp (mm):    {np.round(p_scalp_f, 3)}")
    print(f"Measured scalp offset: {actual_offset:.3f} mm")
    print(f"Outward normal:        {np.round(n_out_f, 6)}")
    print("matsimnibs:\n", np.array2string(mat_fixed, precision=6, suppress_small=True))
    verify_matsimnibs(mat_fixed, head_mask, inv_affine, label="fixed")
    print(f"Saved: {out_fixed}")

    print("\nRun C++ solver with either file:")
    print(f"  ./tms_gmg_solver {args.nifti} <coil.ccd> out_efield.nii.gz --matsimnibs {out_target}")
    print(f"  ./tms_gmg_solver {args.nifti} <coil.ccd> out_efield.nii.gz --matsimnibs {out_fixed}")


if __name__ == "__main__":
    main()