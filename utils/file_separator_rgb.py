import os
import glob
from pathlib import Path

files = glob.glob('splits/*.txt')
classes = sorted(['smile', 'stand', 'wave', 'walk', 'run', 'sit'])
splits = {x: [] for x in classes}

def retrieve_filename(splits, key):
    if local_file.find(key) != -1:
        with open(local_file, 'r') as f:
            for line in f.readlines():
                splits[key] += [line.strip().split()]
    return splits
      
def organize_file(filename, filename_class, filename_split):
    if filename_split is None:
        return
    try:
        Path("dataset/"+filename_class+'/'+filename_split).mkdir(parents=True, exist_ok=True)
        os.rename(
            'dataset/'+filename_class+'/'+filename, 
            "dataset/"+filename_class+'/'+filename_split+'/'+filename)
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
        if entry[1] == '1':
            filename_split = 'training' 
            training += 1
        elif entry[1] == '2': 
            filename_split = 'test'
            test += 1
        else:
            filename_split = None
        organize_file(entry[0], key, filename_split)
print(f'Total samples: {training+test} Training: {training} Test: {test} Ratio: {test/(training+test)}')