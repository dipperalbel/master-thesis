import os
import glob
import shutil
from pathlib import Path

files = glob.glob('splits/*.txt')
classes = sorted(['smile', 'stand', 'wave', 'walk', 'run', 'sit'])
splits = {x: [] for x in classes}

def retrieve_filename(splits, key):
    if local_file.find(key) != -1:
        with open(local_file, 'r') as f:
            for line in f.readlines():
                string_list = line.strip().split()
                string_list[0] = string_list[0][0:-4]
                splits[key] += [string_list]
    return splits
      
def organize_file(filename, filename_class, filename_split):
    if filename_split is None:
        return
    try:
        Path("flow_dataset/"+filename_class+'/'+filename_split).mkdir(parents=True, exist_ok=True)
        shutil.move(
            'flow_dataset/flow/'+filename, 
            "flow_dataset/"+filename_class+'/'+filename_split+'/'+filename)
    except FileNotFoundError:
        print('File '+filename+' not found')


# Append files
for local_file in files:
    for key in splits:
        splits = retrieve_filename(splits, key)

training = 0
test = 0
for key in splits:
    for entry in splits[key]:
        filename = glob.glob('flow_dataset/flow/'+entry[0]+'*/')
        if filename != []:       
            filename = filename[0].split('/')[2] 
            if entry[1] == '1':
                filename_split = 'training' 
                training += 1
            elif entry[1] == '2': 
                filename_split = 'test'
                test += 1
            else:
                filename_split = None
            organize_file(filename, key, filename_split)
print(f'Total samples: {training+test} Training: {training} Test: {test} Ratio: {test/(training+test)}')