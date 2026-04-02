# StereoDislo
A python package to reconstruct dislocations in 3D from 2 images taken at different angle of view

## Reconstruction
```stereorecontruction.py``` aims to reconstruct dislocations in 3 dimension by using two images taken at two angle of view. The reconstruction is obtained through triangulation calculation.
### input
Located in data, two kind of input are needed :
 - raw data files are image text of 2D dislocations. Each dislocation has two images taken at a different angle. In ```stereorecontruction.py``` the origin is the position of the invariant point contained into the tilt axis.
 - ```start_end.csv``` file contains the position of the beginning and the end of each dislocation curve.
### output
Located in results, the outputs consists in numpy arrays of 3D points forming the reconstructed dislocations. The reconstruction is displayed on a 3D plot.
