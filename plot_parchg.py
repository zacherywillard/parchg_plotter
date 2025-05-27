import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_out(out, verbose=False):
    start = out.find('[')
    end = out.find(']')
    
    if start == -1 or end == -1:
        raise ValueError("No position list found in the file.")
    
    position_str = out[start + 1:end]
    position = [float(x.strip()) for x in position_str.split(',')]
    
    if verbose:
        print(f"Extracted position: {position}")
    
    return position

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
    parser.add_argument("-m", "--make_out", type=str, required=True, help="File with defect information output by make_defect.py (e.g., make_out.txt)")
    parser.add_argument("-p", "--parchg", type=str, default="PARCHG", help="Path to PARCHG file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("-o", "--output", type=str, default="psi2_plot.png", help="Filename to save the plot (e.g., plot.png)")

    args = parser.parse_args()

    if args.verbose:
        print("Reading defect position from:", args.make_out)

    with open(args.make_out, 'r') as f:
        line = f.readline().strip()
    defect_position = read_out(line, verbose=args.verbose)
    frac_z_defect = defect_position[2]

    if args.verbose:
        print(f"Fractional z-position of defect: {frac_z_defect:.6f}")
        print("Reading PARCHG file:", args.parchg)

    chg_data, c_length = parse_parchg(args.parchg, verbose=args.verbose)
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
    plt.show()

if __name__ == "__main__":
    main()
