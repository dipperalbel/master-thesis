import os
import glob
from pathlib import Path
import re 

# file_type =  'rgb'
file_type = 'flow'
# dataset = 'hmdb-51'
dataset = 'ucf-101'

split_location = 'ucf101_splits'
test_files = [
    'dann_datasets/'+split_location+'/testlist01.txt']
    
    

train_files = [
    'dann_datasets/'+split_location+'/trainlist01.txt']
classes = sorted(os.listdir('dann_datasets/'+file_type+'/'+dataset+'/x'))
print(classes)
splits = {x: [] for x in classes}
splits_number = {x: {'1': 0, '2': 0} for x in classes}

def retrieve_filename(local_file, splits, test=True):
    with open(local_file, 'r') as f:
        for line in f.readlines():
            line = re.split('[/ ]', line.strip())
            class_name = line[0]
            filename = line[1]
            if class_name in classes:
                filetype = '2' if test else '1'
                splits[class_name].append([filename, filetype])
                if class_name == 'HorseRiding':
                    print(splits[class_name][-1])
                splits_number[class_name][filetype] += 1
    return splits

   
def organize_file(filename, filename_class, filename_split):
    if filename_split is None:
        return
    try:
        Path('dann_datasets/'+file_type+'/'+dataset+'/x/'+filename_class+'/'+filename_split).mkdir(parents=True, exist_ok=True)
        os.rename(
            'dann_datasets/'+file_type+'/'+dataset+'/x/'+filename_class+'/'+filename, 
            'dann_datasets/'+file_type+'/'+dataset+'/x/'+filename_class+'/'+filename_split+'/'+filename)
    except FileNotFoundError:
        # print('File '+filename+' not found')
        pass


# Append files
for local_file in test_files:
    print(local_file)
    splits = retrieve_filename(local_file, splits, test=True)

for local_file in train_files:
    print(local_file)
    splits = retrieve_filename(local_file, splits, test=False)
    
print(splits_number)

training = 0
test = 0
total = 0
none_count = 0
for key in splits:
    for entry in splits[key]:
        total += 1
        # print(entry[1])
        if entry[1] == '1':
            filename_split = 'training' 
            training += 1
        elif entry[1] == '2': 
            filename_split = 'test'
            test += 1
        else:
            none_count += 1
            filename_split = 'lost'
        organize_file(entry[0][0:-4], key, filename_split)
print(f'Total samples: {total} Training: {training} Test: {test} Lost: {none_count} Ratio: {test/(training+test)}')