import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

from unit_conversion import cpd_to_gg, gg_to_mbqvolumem3

def read_data(filename, tree_names):
    data = {}
    with uproot.open(filename) as file:
        for tree_name in tree_names:
            data[tree_name] = file[tree_name][tree_name].array(library="np")
    df = pd.DataFrame(data)
    return df

def event_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def select_prompt_and_delay(df, prompt_E_min, prompt_E_max, delay_E_min, delay_E_max, FV_r_min, FV_r_max, FV_z_min, FV_z_max, dt_min, dt_max, dr_max):
    df['r'] = np.sqrt(df['recX']**2 + df['recY']**2)
    prompt_mask = (
        (df['Evis'] >= prompt_E_min) & (df['Evis'] <= prompt_E_max) &
        (df['r'] >= FV_r_min) & (df['r'] <= FV_r_max) &
        (df['recZ'] >= FV_z_min) & (df['recZ'] <= FV_z_max)
    )
    prompt_events = df[prompt_mask].copy()
    print(prompt_events)

    delay_mask = (
        (df['Evis'] >= delay_E_min) & (df['Evis'] <= delay_E_max) &
        (df['r'] >= FV_r_min) & (df['r'] <= FV_r_max) &
        (df['recZ'] >= FV_z_min) & (df['recZ'] <= FV_z_max)
    )
    delay_events = df[delay_mask].copy()
    print(delay_events)

    # matched_prompt_events = []
    # for i, prompt_event in tqdm(prompt_events.iterrows(), total=len(prompt_events), desc="Processing prompt events"):
    #     for j, delay_event in delay_events.iloc[i+1:].iterrows():
    #         dt = int(delay_event['rec_time']) - int(prompt_event['rec_time'])
    #         if dt_min <= dt <= dt_max:
    #             dr = event_distance(
    #                 prompt_event['recX'], prompt_event['recY'], prompt_event['recZ'],
    #                 delay_event['recX'], delay_event['recY'], delay_event['recZ']
    #             )
    #             if dr <= dr_max:
    #                 prompt_event['dt'] = dt
    #                 prompt_event['dr'] = dr
    #                 matched_prompt_events.append(prompt_event)
    #                 print(matched_prompt_events)

    # matched_prompt_df = pd.DataFrame(matched_prompt_events)

    # prompt_count = len(matched_prompt_df)
    # print(f"满足条件的prompt信号个数: {prompt_count}")
    # return matched_prompt_df, prompt_count
    
    prompt_events['key'] = 1
    delay_events['key'] = 1
    merged = pd.merge(prompt_events, delay_events, on='key', suffixes=('_prompt', '_delay')).drop('key', axis=1)
    merged['dt'] = merged['rec_time_delay'] - merged['rec_time_prompt']
    merged['dr'] = np.sqrt(
        (merged['recX_prompt'] - merged['recX_delay'])**2 +
        (merged['recY_prompt'] - merged['recY_delay'])**2 +
        (merged['recZ_prompt'] - merged['recZ_delay'])**2
    )
    selected = merged[(merged['dt'] >= dt_min) & (merged['dt'] <= dt_max) & (merged['dr'] <= dr_max)]
    print(selected)
    prompt_count = len(selected)
    print(f"满足条件的prompt信号个数: {prompt_count}")
    return selected, prompt_count

def calculate_event_rate(df, prompt_count):
    total_time = df['File_time'].max()
    event_rate = prompt_count / total_time if total_time > 0 else 0
    event_rate_error = np.sqrt(prompt_count) / total_time
    print(f"事例率: {event_rate} 事件/单位时间 ± {event_rate_error}")
    return event_rate, event_rate_error

def main():
    filename = "/junofs/users/njulishuo/OSIRIS/Processed_data/08/OSIRISData_hybrid_20240801_092026_OSIRIS_run-5_2024081_091951_rs_processed.root"
    tree_names = ["Evis", "recX", "recY", "recZ", "rec_time", "File_time"]
    df = read_data(filename, tree_names)
    num_bins = 500
    min_val = 0.38
    max_val = 0.65
    x_min = 0.0
    x_max = 3.0

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
    half_life = 4.458e9 # 'year'
    molar_m = 238.028910 # 'g/mol'
    m_total = 16.202e6 # 'g'
    volume = 20 # 'm^3'
    rho = 860 # 'kg/m^3'

    event_gg = cpd_to_gg(event_rate_cpd, m_total, half_life, molar_m)
    event_gg_error = cpd_to_gg(event_rate_cpd_error, m_total, half_life, molar_m)

    event_mBq_20m3 = gg_to_mbqvolumem3(event_gg, half_life, molar_m, volume, rho)
    event_mBq_20m3_error = gg_to_mbqvolumem3(event_gg_error, half_life, molar_m, volume, rho)

    print(f"事例率: {event_mBq_20m3} ± {event_mBq_20m3_error} mBq/20m3")

if __name__ == "__main__":
    main()
