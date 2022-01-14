import os
import glob
from pathlib import Path

from class_dict import hmdb_2_full
file_type =  'rgb'
# file_type = 'flow'
dataset = 'hmdb-51/'
axis = ''

split_location = 'hmdb51_splits'
files = glob.glob('dann_datasets/'+split_location+'/*.txt')
classes = list(hmdb_2_full.keys())
splits = {x: [] for x in classes}

def retrieve_filename(splits, filename, key):
    with open(filename, 'r') as f:
        for line in f.readlines():
            splits[key] += [line.strip().split(' ')]
    return splits

def retrieve_ucf_filename(splits, key):
    if local_file.find(key) != -1:
        with open(local_file, 'r') as f:
            for line in f.readlines():
                splits[key] += [line.strip().split()]
    return splits
    
def organize_file(filename, filename_class, filename_split):
    if filename_split is None:
        return
    try:
        original_path = 'dann_datasets/'+file_type+'/'+dataset+axis+filename_class+'/'+filename
        new_path = 'dann_datasets/'+file_type+'/'+dataset+axis+filename_class+'/'+filename_split+'/'+filename
        print(original_path, new_path)
        # Path(original_path).mkdir(parents=True, exist_ok=True)
        os.rename(
            original_path, 
            new_path)
    except FileNotFoundError:
        # print('File '+filename+' not found')
        pass


# Append files
for class_name in classes:
    files = sorted(glob.glob('dann_datasets/'+split_location+'/'+class_name+'_test_split*.txt'))
    print(files)
    for file_name in files:
        # print(local_file, key)
        splits = retrieve_filename(splits, file_name, class_name)

training = 0
test = 0
total = 0
none_count = 0

lost_dict = {x: {'train': 0, 'test': 0, 'lost': 0} for x in classes}

for key in splits:
    print(key)
    for entry in splits[key]:
        total += 1
        # print(entry)
        if entry[1] == '1':
            filename_split = 'training' 
            training += 1
            lost_dict[key]['train'] += 1
        elif entry[1] == '2': 
            filename_split = 'test'
            test += 1
            lost_dict[key]['test'] += 1
        else:
            if entry[1] != '0':
                print(entry[1])
            none_count += 1
            filename_split = 'lost'
            lost_dict[key]['lost'] += 1
        # print(entry, key, filename_split)
        # print(entry[0][0:-4])
        organize_file(entry[0][0:-4], key, filename_split)
print(f'Total samples: {total} Training: {training} Test: {test} Lost: {none_count} Ratio: {test/(training+test)}')
print(lost_dict)