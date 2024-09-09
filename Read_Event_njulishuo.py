import numpy as np
import os
import pandas as pd
import uproot
from tqdm import tqdm
import argparse

def process_branch(branch_data):
    if isinstance(branch_data, np.ndarray):
        if branch_data.dtype == 'O':
            return [process_branch(item) for item in branch_data]
        else:
            return branch_data
    elif isinstance(branch_data, list):
        return [process_branch(item) for item in branch_data]
    return branch_data


def read_single_tree(input_path, tree_name):
    with uproot.open(input_path) as file:
        tree = file[tree_name]
        branches = {}
        for branch_name in tqdm(tree.keys(), desc="Reading tree: {}".format(tree_name)):
            try:
                branch_data = tree[branch_name].array(library="np")
                print(f"Branch: {branch_name}, Data Type: {type(branch_data)}")
                if isinstance(branch_data, (np.int_, np.float64, np.bool_, np.ndarray)):
                    branches[branch_name] = branch_data
                elif hasattr(branch_data, "__len__") and not isinstance(branch_data, str):
                    processed_data = [process_branch(sub_item) for sub_item in branch_data]
                    max_length = max(len(item) for item in processed_data if hasattr(item, "__len__"))
                    branches[branch_name] = [item + [None] * (max_length - len(item)) if hasattr(item, "__len__") else item for item in processed_data]
                else:
                    branches[branch_name] = branch_data
            except Exception as e:
                print(f"Skipping branch {branch_name} due to error: {e}")
        df = pd.DataFrame(branches)
        return df


def read_data(input_path, tree_name1, tree_name2):
    if os.path.isfile(input_path) and not os.path.isdir(input_path):
        df1 = read_single_tree(input_path, tree_name1)
        df2 = read_single_tree(input_path, tree_name2)
        df2 = df2.rename(columns={"evtIDx": "evtIDx_cluster"})
        df = pd.concat([df1, df2], axis=1)
        return df
    elif os.path.isdir(input_path):
        root_files = sorted([f for f in os.listdir(input_path) if f.endswith('.root')])
        if not root_files:
            print("No ROOT files found in the specified directory.")
            return None
        dataframes = []
        entry_offset = 0
        for root_file in root_files:
            file_path = os.path.join(input_path, root_file)
            df1 = read_single_tree(file_path, tree_name1)
            df2 = read_single_tree(file_path, tree_name2)
            if df1 is not None and df2 is not None:
                df = pd.concat([df1, df2], axis=1)
                df.index += entry_offset
                entry_offset += len(df)
                dataframes.append(df)
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            return combined_df
        else:
            return None
    else:
        print("Invalid input path. Please provide a valid file or directory path.")
        return None


def calculate_fired_pmt(pmt_ids):
    onePMThits = np.zeros(76, dtype=int)
    for pmt_id in pmt_ids:
        onePMThits[pmt_id] += 1
    fired_pmt_count = np.sum(onePMThits > 0)
    return fired_pmt_count


def process_data(df):
    df['Time_stamp'] = df['n_sec'] * 1e9 + df['n_nsec']
    File_time = (df['Time_stamp'].max() - df['Time_stamp'].min()) / 1.0e9
    Time_weight = 1.0 / File_time

    df_filtered = df.copy()
    df_filtered = df_filtered[(df['muonTag'] == False) & (df['deltaTLSMuon'] > 1e6) & (df['deltaTMuon'] > 1e6) & (df['clusterCharge'].apply(len) >= 1)]
    df_filtered['Evis'] = df_filtered['clusterCharge'].apply(lambda x: sum(x) / 436.0)
    df_filtered['FiredPMT'] = df_filtered['IDhit_pmtId'].apply(calculate_fired_pmt)
    if 'cbfRecVertex' in df.columns:
        df_filtered['recX'] = df_filtered['cbfRecVertex/cbfRecVertex.fCoordinates.fX']
        df_filtered['recY'] = df_filtered['cbfRecVertex/cbfRecVertex.fCoordinates.fY']
        df_filtered['recZ'] = df_filtered['cbfRecVertex/cbfRecVertex.fCoordinates.fZ']
    else:
        df_filtered['recX'] = df_filtered['recPos/recPos.fCoordinates.fX']
        df_filtered['recY'] = df_filtered['recPos/recPos.fCoordinates.fY']
        df_filtered['recZ'] = df_filtered['recPos/recPos.fCoordinates.fZ']

    processed_data = {
        'rec_time': [],
        'recX': [],
        'recY': [],
        'recZ': [],
        'Evis': [],
        'Multi_cluster_check': [],
        'FiredPMT': []
    }

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing Rows"):
        for j in range(len(row['clusterCharge'])):
            processed_data['FiredPMT'].append(int(row['FiredPMT']))
            processed_data['rec_time'].append(row['recT'][j])
            processed_data['recX'].append(row['recX'][j])
            processed_data['recY'].append(row['recY'][j])
            processed_data['recZ'].append(row['recZ'][j])
            processed_data['Evis'].append(row['Evis'])
            if j >= 1:
                processed_data['Multi_cluster_check'].append(True)
                if len(processed_data['Multi_cluster_check']) > 1: 
                    processed_data['Multi_cluster_check'][-2] = True
            else:
                processed_data['Multi_cluster_check'].append(False)

    processed_data['rec_time'] = np.array(processed_data['rec_time'])
    processed_data['recX'] = np.array(processed_data['recX'])
    processed_data['recY'] = np.array(processed_data['recY'])
    processed_data['recZ'] = np.array(processed_data['recZ'])
    processed_data['Evis'] = np.array(processed_data['Evis'])
    processed_data['FiredPMT'] = np.array(processed_data['FiredPMT'])
    
    actual_time_range = (processed_data['rec_time'].max() - processed_data['rec_time'].min()) / 1e9 
    if abs(actual_time_range - File_time) >= 1:
        print(f'警告: 时间差距超出范围！实际时间范围: {actual_time_range:.2f}秒，期望时间范围: {File_time:.2f}秒')   
    else:
        meancharge = np.mean(processed_data['Evis'])
        print(f'Mean charge: {meancharge:.6f}')

    processed_df = pd.DataFrame({
        'rec_time': processed_data['rec_time'],
        'recX': processed_data['recX'],
        'recY': processed_data['recY'],
        'recZ': processed_data['recZ'],
        'Evis': processed_data['Evis'],
        'Multi_cluster_check': processed_data['Multi_cluster_check'],
        'FiredPMT': processed_data['FiredPMT'],
        'weight': np.ones(len(processed_data['rec_time'])) * Time_weight,
        'File_time': np.ones(len(processed_data['rec_time'])) * File_time,
    })
    return processed_df


def save_to_root(df, file_path):
    with uproot.recreate(file_path) as f:
        for column in tqdm(df.columns, desc="Saving Columns"):
            f[column] = {column: df[column].to_numpy()}


def main():
    print("Welcome to njulishuo Event Reconstruction program.")

    input_folder = "/junofs/users/njulishuo/OSIRIS/Raw_data/08/"
    output_folder = "/junofs/users/njulishuo/OSIRIS/Processed_data/08/"
    processed_log = os.path.join(output_folder, "processed_files.txt")
    error_files = []

    if os.path.exists(processed_log):
        with open(processed_log, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    tree_name_01 = "cluster_reco"
    tree_name_02 = "recoTree"

    for root_file in sorted(os.listdir(input_folder)):
        if root_file.endswith('.root') and root_file not in processed_files:
            your_path = os.path.join(input_folder, root_file)
            print(your_path)
            save_path = os.path.join(output_folder, root_file.replace('.root', '_processed.root'))
            
            try:
                df = read_data(your_path, tree_name_01, tree_name_02)
                if df is not None:
                    process_df = process_data(df)
                    save_to_root(process_df, save_path)
                    with open(processed_log, "a") as f:
                        f.write(root_file + "\n")
                    print(f"处理完成: {root_file}")
                else:
                    print(f"未能处理数据: {root_file}")
            except Exception as e:
                print(f"处理文件 {root_file} 时出错: {e}")
                error_files.append(root_file)
                continue

    if error_files:
        log_file = os.path.join(output_folder, "error_log.txt")
        with open(log_file, "w") as log:
            log.write("以下文件处理时出错:\n")
            for error_file in error_files:
                log.write(f"{error_file}\n")
        print(f"错误日志已保存到 {log_file}")

if __name__ == "__main__":
    main()
