# COIL POSITIONS

## MATSIMNIBS.TXT FILE

A matsimnibs.txt file has the following structure:
```math
\begin{bmatrix} pX1 & pY1 & pZ1 & pC1 \\ pX2 & pY2 & pZ2 & pC2 \\ pX3 & pY3 & pZ3 & pC3 \\ 0 & 0 & 0 & 1 \end{bmatrix}
```
Basically, in simple terms, matsimnibs defines an affine transformation that handles coil center position and coil handle rotation effectively. For further information, please refer to SimNIBS documentation on how are these numbers defined and calculated.