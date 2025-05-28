import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_def_pos(file="POSCAR", verbose=False):
    """
    Reads a position line (e.g., 0.3 0.3 0.3) from the top of a POSCAR file.

    Parameters:
        file (str): Path to the POSCAR file. Defaults to 'POSCAR'.
        verbose (bool): If True, prints the extracted position.

    Returns:
        list: A list of three floats representing the position.
    """
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    position = [float(x) for x in parts]
                    if verbose:
                        print(f"Extracted position: {position}")
                    return position
                except ValueError:
                    continue  # Ignore lines that can't be parsed as floats

    raise ValueError("No valid position line found at the top of the POSCAR file.")

def parse_parchg(file_path, verbose=False):
    with open(file_path, "r") as f:
        lines = f.readlines()

    c_vector = np.array(list(map(float, lines[4].split())))
    c_length = np.linalg.norm(c_vector)
    
    if verbose:
        print(f"c vector: {c_vector}")
        print(f"c length: {c_length:.4f} Å")

    grid_indices = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            grid_indices.append((i, list(map(int, parts))))
    grid_line_index, (nx, ny, nz) = grid_indices[-1]

    if verbose:
        print(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")

    data_lines = lines[grid_line_index + 1:]
    data = np.array([float(x) for x in " ".join(data_lines).split()])
    chg_data = data.reshape((nx, ny, nz), order='F')

    return chg_data, c_length

def main():
    parser = argparse.ArgumentParser(description="Plot psi^2 vs z from a PARCHG file.")
    parser.add_argument("-p", "--pos", type=str, default="POSCAR", help="File with defect information output by make_defect.py (e.g., make_out.txt)")
    parser.add_argument("-f", "--file", type=str, default="PARCHG", help="Path to PARCHG file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("-o", "--output", type=str, default="psi2_plot.png", help="Filename to save the plot (e.g., plot.png)")

    args = parser.parse_args()

    if args.verbose:
        print("Reading defect position from:", args.pos)

    defect_position = read_def_pos(args.pos, verbose=args.verbose)
    frac_z_defect = defect_position[2]

    if args.verbose:
        print(f"Fractional z-position of defect: {frac_z_defect:.6f}")
        print("Reading PARCHG file:", args.file)
        print("Reading POSCAR file:", args.pos)
        print("Extracted defect position:", defect_position)

    chg_data, c_length = parse_parchg(args.file, verbose=args.verbose)
    psi2_z = np.mean(np.mean(chg_data, axis=0), axis=0)
    nz = chg_data.shape[2]
    z_vals = np.linspace(0, c_length, nz)
    z_defect = frac_z_defect * c_length
    z_vals_shifted = z_vals - z_defect

    if args.verbose:
        print(f"Shifting z-axis to align defect at z = 0.0 Å (z_defect = {z_defect:.4f} Å)")

    plt.figure(figsize=(4, 6))
    plt.plot(psi2_z, z_vals_shifted)
    plt.xlabel(r'$\psi^2$')
    plt.ylabel('z (Å)')
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    if args.verbose:
        print("Plot saved as: psi2_plot.png")

if __name__ == "__main__":
    main()
