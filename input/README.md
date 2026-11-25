# INPUT

## CHARM SEGMENTED FINAL TISSUE FILE

Currently, the C++ MCGC solver supports charm segmented final tissue file in MNI-space or patient-space. A standard charm segemented final tissue has the same dimension as the original MRI-T1w file and has 11 numbers from 0 to 10 to label each voxels in the file. 

## CONTINOUS CONDUCTIVITY VOLUME FILE

In the future, the C++ MCGC solver will have minor adjustments to support continuous conductivity file in MNI-space or patient-space. These files are the direct out of a Residual Neural Network (ResNet) or any Deep Learning algorithm that maps MRI-T1w volumetric space to conductivity space.