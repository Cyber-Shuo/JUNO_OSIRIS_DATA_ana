import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm
from datetime import datetime
import csv

from BiPo214_cut import read_data, select_prompt_and_delay, event_distance, calculate_event_rate
from unit_conversion import cpd_to_gg, gg_to_mbqvolumem3

def load_processed_files(csv_file):
    if not os.path.exists(csv_file):
        return set()
    processed_files = set()
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                processed_files.add(row[0])
    return processed_files

def save_to_csv(csv_file, filename, file_time, event_mBq_20m3, event_mBq_20m3_error):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([filename, file_time, event_mBq_20m3, event_mBq_20m3_error])

def log_error_file(txt_file, filename, error_message):
    with open(txt_file, 'a') as f:
        f.write(f"Error processing file: {filename}\n")
        f.write(f"Error message: {error_message}\n\n")

def process_files_in_folder(folder_path, csv_file, error_file):
    file_times = []
    event_rate_list, event_mBq_20m3_list = [], []
    event_rate_error_list, event_mBq_20m3_error_list = [], []

    processed_files = load_processed_files(csv_file)

    root_files = [filename for filename in os.listdir(folder_path) if filename.endswith("rs_processed.root")]

    for filename in tqdm(root_files, desc="Processing files", unit="file"):
        if filename in processed_files:
            print(f"Skipping already processed file: {filename}")
            continue
        filepath = os.path.join(folder_path, filename)
        try:
            print(f"Processing file: {filename}")

            tree_names = ["Evis", "recX", "recY", "recZ", "rec_time", "File_time"]
            df = read_data(filepath, tree_names)

            time_str = filename.split("_")[2] + filename.split("_")[3]
            file_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
            file_times.append(file_time)

            prompt_E_min, prompt_E_max = 0.3, 3.3
            delay_E_min, delay_E_max = 0.75, 1.15    
            FV_r_min, FV_r_max = 0, 200
            FV_z_min, FV_z_max = -200, 200
            dt_min, dt_max = 1e3, 1.5e6
            dr_max = 150

            select_df, prompt_count = select_prompt_and_delay(df, prompt_E_min, prompt_E_max, delay_E_min, delay_E_max, FV_r_min, FV_r_max, FV_z_min, FV_z_max, dt_min, dt_max, dr_max)
            event_rate, event_rate_error = calculate_event_rate(df, prompt_count)

            event_rate_cpd = event_rate * 24 * 60 * 60
            event_rate_cpd_error = event_rate_error * 24 * 60 * 60
            half_life = 4.458e9  # 'year'
            molar_m = 238.028910  # 'g/mol'
            m_total = 16.202e6  # 'g'
            volume = 20  # 'm^3'
            rho = 860  # 'kg/m^3'

            event_gg = cpd_to_gg(event_rate_cpd, m_total, half_life, molar_m)
            event_gg_error = cpd_to_gg(event_rate_cpd_error, m_total, half_life, molar_m)

            event_mBq_20m3 = gg_to_mbqvolumem3(event_gg, half_life, molar_m, volume, rho)
            event_mBq_20m3_error = gg_to_mbqvolumem3(event_gg_error, half_life, molar_m, volume, rho)

            event_mBq_20m3_list.append(event_mBq_20m3)
            event_mBq_20m3_error_list.append(event_mBq_20m3_error)

            save_to_csv(csv_file, filename, file_time, event_mBq_20m3, event_mBq_20m3_error)

        except Exception as e:
            error_message = str(e)
            log_error(error_file, filename, error_message)
            print(f"Error processing file: {filename}, Error: {error_message}")
            continue

    return file_times, event_mBq_20m3_list, event_mBq_20m3_error_list

def plot_event_rate_evolution(file_times, event_rate_list, event_rate_error_list):
    plt.figure(figsize=(10, 6))
    plt.errorbar(file_times, event_rate_list, yerr=event_rate_error_list, fmt='o', label="U238 Event Rate", ecolor='red', capsize=3)
    plt.xlabel("Time")
    plt.ylabel("U238 Event Rate (events/unit time)")
    plt.title("U238 Event Rate Evolution Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

def main():
    folder_path = "/junofs/users/njulishuo/OSIRIS/Processed_data/08/"
    csv_file = "/junofs/users/njulishuo/OSIRIS/ana/U238ana/processed_files.csv"
    error_file = "/junofs/users/njulishuo/OSIRIS/ana/U238ana/error_log.txt" 
    file_times, event_mBq_20m3_list, event_mBq_20m3_error_list = process_files_in_folder(folder_path, csv_file, error_file)
    print(file_times, event_mBq_20m3_list, event_mBq_20m3_error_list)
    # plot_event_rate_evolution(file_times, event_rate_list, event_rate_error_list)

if __name__ == "__main__":
    main()
