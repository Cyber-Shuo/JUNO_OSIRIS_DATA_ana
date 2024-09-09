import uproot
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename, tree_name):
    with uproot.open(filename) as file:
        tree = file[tree_name]
        data = tree[tree_name].array(library="np")
    return data

def plot_heatmap(x, y, z, evis, output_file_prefix):
    plt.figure(figsize=(8, 6))
    x = x * 10
    y = y * 10
    plt.scatter(x, y, c=evis, cmap='viridis', marker='.', edgecolor='none', s=1)
    plt.colorbar(label='Evis')
    plt.xlabel('X(mm)', size = '20')
    plt.ylabel('Y(mm)', size = '20')
    plt.title('Scatter Plot of X-Y Plane', size = '20')
    plt.savefig(f'{output_file_prefix}_xy.pdf')
    plt.show()

    plt.figure(figsize=(8, 6))
    r = np.sqrt(x**2 + y**2)
    z = z * 10
    plt.scatter(r, z, c=evis, cmap='viridis', marker='.', edgecolor='none', s=1)
    plt.colorbar(label='Evis')
    plt.xlabel('R(mm)', size = '20')
    plt.ylabel('Z(mm)', size = '20')
    plt.title('Scatter Plot of R-Z Plane', size = '20')
    plt.savefig(f'{output_file_prefix}_rz.pdf')
    plt.show()

def main():
    filename = "/junofs/users/njulishuo/OSIRIS/Processed_data/data_20240811/OSIRISData_hybrid_20240811_161147_OSIRIS_run-5_20240811_161029_rs_processed.root"
    output_file_prefix = "OSIRISData_hybrid_20240811_161147_OSIRIS_run-5_20240811_161029_rs_processed"
    rec_x = read_data(filename, "recX")
    rec_y = read_data(filename, "recY")
    rec_z = read_data(filename, "recZ")
    evis = read_data(filename, "Evis")

    plot_heatmap(rec_x, rec_y, rec_z, evis, output_file_prefix)

if __name__ == "__main__":
    main()
