import numpy as np
import nibabel as nib

seg_img = nib.load("input/final_tissues_MNI.nii.gz")
ef_img = nib.load("output_efield.nii.gz")

seg = np.squeeze(np.rint(seg_img.get_fdata()).astype(int))
ef = np.squeeze(ef_img.get_fdata())

# CHARM labels: 1 = WM, 2 = GM, 3 = CSF, 5 = Scalp
for name, lbl in [("Gray Matter (2)", 2), ("White Matter (1)", 1), ("Scalp (5)", 5)]:
    vals = ef[seg == lbl]
    if vals.size:
        print(f"{name:18s} | max = {vals.max():.3f} V/m | p99.9 = {np.percentile(vals, 99.9):.3f} V/m")

# MNI coordinates of the peak inside Gray Matter
gm_ef = np.where(seg == 2, ef, -1.0)
peak_vox = np.unravel_index(np.argmax(gm_ef), gm_ef.shape)
peak_mni = nib.affines.apply_affine(ef_img.affine, peak_vox)
target = np.array([-38.0, -22.0, 56.0])

print(f"\nGM Peak MNI (mm):  {np.round(peak_mni, 2)}")
print(f"Distance to target: {np.linalg.norm(peak_mni - target):.2f} mm")