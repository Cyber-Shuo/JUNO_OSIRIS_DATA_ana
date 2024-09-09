import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_data(filename, tree_name):
    with uproot.open(filename) as file:
        tree = file[tree_name]
        data = tree[tree_name].array(library="np")
    return data

def fit_function(x, a, mean, sigma, c0, c1):
    gauss = a * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    polynomial = c0 + c1 * x
    return gauss + polynomial

def plot_and_fit(evis_data, num_bins, min_val, max_val, x_min, x_max, save_path):
    if evis_data.size == 0:
        print("Error: No data to plot.")
        return None, None, None

    hist, bin_edges = np.histogram(evis_data, bins=num_bins, range=(x_min, x_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = (bin_centers >= min_val) & (bin_centers <= max_val)
    fit_bin_centers = bin_centers[mask]
    fit_hist = hist[mask]

    try:
        popt, pcov = curve_fit(fit_function, fit_bin_centers, fit_hist, p0=[1000, 0.5, 0.05, 500, -1000])
    except RuntimeError as e:
        print(f"Error: Fit failed with error {e}")
        return None, None, None

    plt.figure(figsize=(12, 8))
    plt.errorbar(bin_centers, hist, yerr=np.sqrt(hist), fmt='.', markersize=2, elinewidth=0.5, label='Evis', color='black')
    plt.plot(fit_bin_centers, fit_function(fit_bin_centers, *popt), label='Gaussian + Polynomial Fit', color='red')

    gauss = popt[0] * np.exp(-0.5 * ((fit_bin_centers - popt[1]) / popt[2]) ** 2)
    polynomial = popt[3] + popt[4] * fit_bin_centers

    plt.plot(fit_bin_centers, gauss, label='Gaussian fit', color='blue')
    plt.plot(fit_bin_centers, polynomial, label='Polynomial fit', color='magenta')

    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('Evis', size = '20')
    plt.ylabel('Frequency log', size = '20')
    plt.title('Evis Distribution with Gaussian + Polynomial Fit', size = '20')
    plt.legend()
    plt.savefig(save_path)

    return popt, gauss, polynomial

def main():
    filename = "/junofs/users/njulishuo/OSIRIS/Processed_data/data_20240811/OSIRISData_hybrid_20240811_161147_OSIRIS_run-5_20240811_161029_rs_processed.root"
    num_bins = 500
    min_val = 0.38
    max_val = 0.65
    x_min = 0.0
    x_max = 3.0
    save_path = "/junofs/users/njulishuo/OSIRIS/Figure/Evis/Evis_fit.pdf"

    evis = read_data(filename, "Evis")
    plot_and_fit(evis, num_bins, min_val, max_val, x_min, x_max, save_path)

if __name__ == "__main__":
    main()
