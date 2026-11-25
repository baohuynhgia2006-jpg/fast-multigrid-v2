# FAST-MULTIGRID++ COMPUTATION

## INTRODUCTION

E-field simulation for Transcranial Magnetic Stimulation (TMS) is an important problem. Medical professionals need accurate E-field data to verify beforehand the placement of TMS coil. Existing E-field simulation pipeline is time-consuming and computationally expensive to conduct. It takes on average 5 minutes or more to compute the E-field, given a coil position and head model. We decides to bridge the gap by incorporating the fast multigrid-based computation proposed by Ilkka Laakso and Akimasa Hirata in their paper "Fast multigrid-based computation of the induced
electric field for transcranial magnetic stimulation".

This is an improved version of an older repository containing the implementation of the fast multigrid algorithm in C++. The older repository is no longer compatible with our uses and faces severe implementation flaws. Hence, we decided to simplify the code and migrate the implementation to this repository. The older pipeline lacks support for MNI-space, matsimnibs cooordinate file and was slower on average.

## GUIDE
1. To build the tms_gmg_solver, use the following command and make sure you have zlib installed in your MYSYS2. If not, you can always use MYSYS2-MinGW64 command prompt to install zlib.

```bash
g++ -std=c++17 -O3 -march=native -fopenmp -DUSE_ZLIB main.cpp -lz -o tms_gmg_solver
```

2. Once everything have been compiled and an .exe file is created. You can use the following command with these flags to run your simulation:

```bash
./tms_gmg_solver path\to\final_tissues_MNI.nii path\to\Magstim_70mm_Fig8.ccd path\to\out_efield.nii --matsimnibs matsimnibs.txt --didt 1e6 --levels 4 --tol 1e-5
```

It is advisable to use MNI-coordinate space NifTi1 files instead of subject space. Conversion between MNI and patient-space can be done before feeding the resulting converted file into the C++ solver. We also recommend the usgae of matsimnibs.txt to input the coil coordinates, which is easier to handle and can be ran on different softwares to verify and cross-check the results. If a matsimnibs.txt file is not supplied, the standard superior position of the coil is used. Please be aware also that we are using the coordinate system as defined by SimNIBS, please refer to SimNIBS documentations for any questions regarding coordinate systems.

## ACKNOWLEDGEMENT

Our algorithm made extensive use of the ideas outlined by the following paper, Laakso, Ilkka; Hirata, Akimasa . (2012). Fast multigrid-based computation of the induced electric field for transcranial magnetic stimulation. Physics in Medicine and Biology, 57(23), 7753–7765.