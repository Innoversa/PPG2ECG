import torch.nn as nn
import torch
import numpy as np
import pdb
import sys
from pathlib import Path
import pandas as pd
from pprint import pprint
import wfdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import multiprocessing as mp
import json
import pickle
import time
from tqdm import tqdm

# Finding potassium items
mimic_v_data_path = "/ssd-shared/physionet.org/files/mimiciv/2.2/"
d_items = pd.read_csv(mimic_v_data_path + "icu/d_items.csv.gz")
print("done reading d_items...")
pdb.set_trace()
potassium_related = []
for item in d_items["label"]:
    if isinstance(item, str):
        if "pota" in item.lower():
            # print(item)
            # print("*"*100)
            potassium_related.append(item)

print("finding potassium related items...")
potasium_items = d_items[d_items["label"].isin(potassium_related)].reset_index(
    drop=True
)
potasium_items.to_csv("./data/Potassium_Measurement.csv")


print("merging potassium measurements and ecg files...")


# merging potassium measurements and ecg files
def change_pid(row):
    return int(row[1:])


print("reading (big!) chartevents file...")

start = time.time()
chart_events = pd.read_csv(
    "/ssd-shared/physionet.org/files/mimiciv/2.2/icu/chartevents.csv.gz"
)

# data = pd.read_csv("/home/grads/z/zhale/MIMICIII/physionet.org/files/mimiciii/1.4/CHARTEVENTS.csv.gz", nrows=1000, compression='gzip',
#                    error_bad_lines=False)

end = time.time()

print(f"done reading chart events in {end - start}")

# potasium_items = pd.read_csv("Potassium_Measurement.csv")

print(potasium_items)


chart_events_w_potassium = chart_events[
    chart_events["itemid"].isin(potasium_items["itemid"].unique())
]
chart_events_w_potassium.to_csv("./data/chart_events_w_potassium.csv", index=False)
# chart_events_w_potassium = pd.read_csv("chart_events_w_potassium.csv")
print("added potassium measurements to chart events...")

ecg_meta_data = pd.read_csv(
    f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/meta_data.csv"
)
print(ecg_meta_data)
ecg_meta_data["subject_id"] = ecg_meta_data["PID"].apply(change_pid)

print("merging ECG with potassium data...")
potassium_w_ecg = pd.merge(chart_events_w_potassium, ecg_meta_data, on=["subject_id"])
print(potassium_w_ecg)
potassium_w_ecg.to_csv("./data/potassium_w_ecg.csv", index=False)
print("almost done... Let's see number of nans")
print(potassium_w_ecg.isna().sum())
print("Finally done!")
