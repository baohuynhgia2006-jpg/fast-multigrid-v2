"""Inspect and validate 4D RAS+ vector E-field NIfTI output from tms_gmg_solver."""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter


def main():
    p = argparse.ArgumentParser(description="Inspect 4D TMS vector E-field NIfTI.")
    p.add_argument("--seg", default="input/final_tissues_MNI.nii.gz", help="CHARM segmentation NIfTI")
    p.add_argument("--vec", default="output_efield_vec.nii.gz", help="4D vector E NIfTI (RAS+)")
    p.add_argument("--mag", default="output_efield.nii.gz", help="Optional 3D scalar |E| NIfTI for consistency check")
    p.add_argument("--matsimnibs", default="matsimnibs_from_target.txt", help="Optional 4x4 matsimnibs matrix file")
    p.add_argument("--target", type=float, nargs=3, default=[-38.0, -22.0, 56.0], metavar=("X", "Y", "Z"),
                   help="Target MNI coordinate in mm (default: -38 -22 56)")
    p.add_argument("--radius", type=float, default=5.0, help="ROI sphere radius in mm (default: 5.0)")
    args = p.parse_args()

    seg_img = nib.load(args.seg)
    vec_img = nib.load(args.vec)

    seg = np.squeeze(np.rint(seg_img.get_fdata())).astype(np.int32)
    E = np.squeeze(vec_img.get_fdata(dtype=np.float32))
    affine = vec_img.affine

    if E.ndim != 4 or E.shape[-1] != 3:
        raise ValueError(f"Expected 4D vector volume (nx, ny, nz, 3), got {E.shape}")
    if E.shape[:3] != seg.shape:
        raise ValueError(f"Spatial shape mismatch: vec {E.shape[:3]} vs seg {seg.shape}")

    E_mag = np.linalg.norm(E, axis=-1)
    head_mask = seg > 0
    air_max = float(E_mag[~head_mask].max()) if np.any(~head_mask) else 0.0

    print("=== 1. Volume & Layout Sanity Check ===")
    print(f"Vector shape / dtype:  {E.shape} / {E.dtype}")
    print(f"Affine diagonal (mm):  {np.round(np.diag(affine[:3, :3]), 4).tolist()}")
    print(f"Max |E| outside head:  {air_max:.6f} V/m")

    # Verify ||E_vec|| matches scalar |E| (catches 4D stride/channel-order or axis bugs)
    if args.mag and Path(args.mag).is_file():
        mag_scalar = np.squeeze(nib.load(args.mag).get_fdata(dtype=np.float32))
        diff = np.abs(E_mag - mag_scalar)
        print(f"||E_vec|| vs |E| file: max_abs_diff = {diff.max():.6e} V/m | p99.9_diff = {np.percentile(diff, 99.9):.6e} V/m")

    print("\n=== 2. Tissue Summary (from ||E_vec||) ===")
    for name, lbl in [("Gray Matter (2)", 2), ("White Matter (1)", 1), ("Scalp (5)", 5)]:
        vals = E_mag[seg == lbl]
        if vals.size:
            print(f"  {name:18s} | max = {vals.max():.3f} V/m | p99.9 = {np.percentile(vals, 99.9):.3f} V/m")

    target = np.array(args.target, dtype=np.float64)
    gm_mag = np.where(seg == 2, E_mag, -1.0)
    peak_vox = np.unravel_index(np.argmax(gm_mag), gm_mag.shape)
    peak_mni = nib.affines.apply_affine(affine, peak_vox)
    peak_vec = E[peak_vox]
    print(f"  GM Peak MNI (mm):  {np.round(peak_mni, 2).tolist()} (dist to target = {np.linalg.norm(peak_mni - target):.2f} mm)")
    print(f"  GM Peak E (RAS+):  {np.round(peak_vec, 4).tolist()} V/m")

    # Estimate inward cortical normal (CSF -> GM/WM) in world RAS+ space
    brain_smooth = gaussian_filter(((seg == 1) | (seg == 2)).astype(np.float32), sigma=1.5)
    grad_vox = np.stack(np.gradient(brain_smooth), axis=-1)
    inv_A_T = np.linalg.inv(affine[:3, :3]).T
    grad_world = np.einsum("ij,...j->...i", inv_A_T, grad_vox)
    grad_norm = np.linalg.norm(grad_world, axis=-1, keepdims=True)
    n_inward = np.divide(grad_world, grad_norm, out=np.zeros_like(grad_world), where=grad_norm > 1e-6)

    E_n = np.sum(E * n_inward, axis=-1)
    E_t = np.sqrt(np.maximum(E_mag**2 - E_n**2, 0.0))

    # Build MNI coordinates only inside a bounding box around the target (fast & low RAM)
    ijk = np.indices(seg.shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(affine, ijk).reshape(*seg.shape, 3)
    dist = np.linalg.norm(xyz - target, axis=-1)
    roi = (seg == 2) & (dist <= args.radius)

    print(f"\n=== 3. Target GM ROI (r={args.radius:.1f} mm around {target.tolist()}, n={int(roi.sum())} voxels) ===")
    if not np.any(roi):
        print("  No GM voxels found in ROI. Check target coordinates or increase --radius.")
        return

    mean_E = E[roi].mean(axis=0)
    mean_dir = mean_E / np.linalg.norm(mean_E)
    print(f"  Mean |E| / max |E|: {E_mag[roi].mean():.3f} / {E_mag[roi].max():.3f} V/m")
    print(f"  Mean E_n (signed):  {E_n[roi].mean():+.3f} V/m (mean |E_n| = {np.abs(E_n[roi]).mean():.3f} V/m)")
    print(f"  Mean E_t:           {E_t[roi].mean():.3f} V/m")
    print(f"  Mean E (RAS+):      {np.round(mean_E, 4).tolist()} V/m")
    print(f"  Unit E direction:   {np.round(mean_dir, 4).tolist()}")

    # Compare against coil +y axis from matsimnibs (both at GM ROI and superficial scalp under coil)
    if args.matsimnibs and Path(args.matsimnibs).is_file():
        mat = np.loadtxt(args.matsimnibs)
        coil_py = mat[:3, 1] / np.linalg.norm(mat[:3, 1])
        coil_pos = mat[:3, 3]

        def report_align(label, vec):
            u = vec / np.linalg.norm(vec)
            dot = float(np.clip(np.dot(u, coil_py), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(abs(dot))))
            print(f"  {label:20s} | angle(E, ±coil_y) = {ang:5.2f} deg | signed dot = {dot:+.4f}")

        print("\n=== 4. Coil Axis Alignment Check ===")
        print(f"  Coil center (mm):   {np.round(coil_pos, 2).tolist()}")
        print(f"  Coil +y axis:       {np.round(coil_py, 4).tolist()}")
        report_align("GM Target ROI", mean_E)

        scalp_roi = (seg == 5) & (np.linalg.norm(xyz - coil_pos, axis=-1) <= 15.0)
        if np.any(scalp_roi):
            report_align("Scalp under coil", E[scalp_roi].mean(axis=0))


if __name__ == "__main__":
    main()