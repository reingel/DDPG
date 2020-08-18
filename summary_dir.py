import os
from datetime import date

def summary_dir():
    today_dir = 'runs/' + str(date.today())

    if os.path.exists(today_dir):
        seq_dirs = os.listdir(today_dir)
        if seq_dirs:
            next_seq = max(map(int, seq_dirs)) + 1
            seq_str = ('0' + str(next_seq))[-2:]
        else:
            seq_str = '01'
    else:
        seq_str = '01'

    return today_dir + '/' + seq_str
