import os
import math
import random

def get_audiofile_paths(dataset_path):
    audiofile_paths = []
    print(f'\nReading audio files from {dataset_path}...')
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            audiofile_paths.append(os.path.join(root, name))
    return audiofile_paths  

def get_train_test(dataset_path):
    audiofile_paths = get_audiofile_paths(dataset_path)
    return audiofile_paths
def write_out_file(train, test, out_path):
    trainout = open(f'{out_path}/train.txt', "w")
    for sfx in train:
        trainout.write(sfx + "\n")
    testout = open(f'{out_path}/test.txt', "w")
    for sfx in test:
        testout.write(sfx + "\n")

if __name__ == '__main__':
    #Put your dataset path here
    DATASET_PATH = ''
    TRAIN_TEST_SPLIT = 0.8
    OUT_PATH = 'filelists'
    
    audiofile_paths = get_train_test(DATASET_PATH)
    if os.path.exists(OUT_PATH) == False:
        os.makedirs(OUT_PATH)
    #Randomise to get random train/test split
    random.shuffle(audiofile_paths)
    total_files = len(audiofile_paths)
    train_split = audiofile_paths[0:math.ceil(total_files*TRAIN_TEST_SPLIT)]
    test_split = audiofile_paths[math.ceil(total_files*TRAIN_TEST_SPLIT):-1]

    write_out_file(train_split, test_split, OUT_PATH)
