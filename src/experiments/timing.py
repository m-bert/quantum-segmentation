import requests
import os
from datetime import datetime
import json

from collections import defaultdict
from enum import Enum

import pandas as pd

class Mode(Enum):
    GRAM_SCHMIDT = "gram_schmidt"
    LANCZOS = "lanczos"

SAPI_HOME = "https://na-west-1.cloud.dwavesys.com/sapi/v2"
SAPI_TOKEN = os.getenv("DWAVE_API_TOKEN")

session = requests.Session()
session.headers = {'X-Auth-Token': SAPI_TOKEN,
                   'Content-type': 'application/json',
                   'Accept': 'application/vnd.dwave.sapi.problem+json; version=3'}

def get_total_time(problem_id):
    r = session.get(f"{SAPI_HOME}/problems/{problem_id}")
    r = r.json()

    qpu_time = r['answer']['timing']['qpu_access_time']
    qpu_time_seconds = qpu_time / 10**6

    start = datetime.fromisoformat(r['submitted_on'])
    end   = datetime.fromisoformat(r['solved_on'])

    elapsed = end - start

    return elapsed.total_seconds(), qpu_time_seconds

def load_and_process_data(img_name, mode,):
    filename = f"{img_name}_{mode.value}_annealer.json"
    path = os.path.join(os.path.dirname(__file__), "results", img_name, filename)

    timing_data = defaultdict()

    with open(path, "r") as f:
        raw_data = json.load(f)

        for M_str, data in raw_data["M"].items():
            M = int(M_str)

            service_time, qpu_time = get_total_time(data["problem_id"])
            
            timing_data[M] = {
                "dwave_call_time": data["dwave_call_time"],
                "service_time": service_time,
                "qpu_time": qpu_time
            }

        return timing_data
    
    return False

if __name__ == "__main__":
    img_name = "7"

    data = load_and_process_data(img_name, Mode.LANCZOS)

    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'M'
    df.reset_index(inplace=True)
    
    for M, timing in data.items():
        df.loc[df['M'] == M, 'dwave_call_time'] = timing['dwave_call_time']
        df.loc[df['M'] == M, 'service_time'] = timing['service_time']
        df.loc[df['M'] == M, 'qpu_time'] = timing['qpu_time']

    filename = f"{img_name}_timing_data.csv"
    file_path = os.path.join(os.path.dirname(__file__), "results", img_name, filename)

    df.to_csv(file_path, index=False, sep=';')