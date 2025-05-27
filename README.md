# PARCHG plotter

## Overview

This script plots the real-space probability density  $|\psi(z)|^2$ along the z-axis from a VASP `PARCHG` file. The z-axis is shifted so that the defect position is centered at  z = 0.

The defect position (in fractional coordinates) is extracted from the output produced by `make_defect.py`.

---

## Features

- Parses a VASP `PARCHG` file to extract  $|\psi|^2$ 
- Reads the fractional defect position from a file
- Aligns the defect at z = 0
- Supports verbose logging
- Saves the plot to a user-specified file

---

## Requirements
- NumPy
- Matplotlib
