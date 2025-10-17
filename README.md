# Dynamic-Contact-Angle-Analysis-of-Droplet-Impact-on-Solid-Surfaces
This project focuses on the experimental and computational analysis of droplet impact on a solid substrate, aiming to determine the dynamic contact angle and other geometric and physical properties of the droplet over time.

Using high-speed imaging and numerical techniques, the study investigates the droplet’s behavior upon impact, deformation, and stabilization on the surface.

The work is divided into two main stages:

Image Processing and Contour Extraction

Processing of high-speed video frames using thresholding and edge detection (Canny).

Alignment of the droplet base to ensure accurate contact reference.

Tracking of the droplet’s center of mass and trajectory over time.

Contact Angle and Geometric Analysis

Fitting of left and right droplet contours using cubic splines and least-squares polynomials.

Measurement of left and right dynamic contact angles at each frame.

Calculation of geometric and physical parameters:

Perimeter asymmetry

Spreading factor (Sf = D_base / H)

Volume (via revolution)

Kinetic energy evolution

These analyses enable the characterization of wetting dynamics, symmetry, and energy dissipation during the droplet–surface interaction, serving as a foundation for further modeling in TP5 and TP6.

Technologies & Tools

Python (OpenCV, NumPy, SciPy, Matplotlib, Pandas)

Image segmentation and contour extraction

Curve fitting (polynomial & spline)

Dynamic angle computation and physical property estimation
