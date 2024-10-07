import json
import random
import os
from PIL import Image


img_pth = {
    'FGVCAircraft': 'data/fgvc_aircraft/images/',
    'Food101': 'data/food-101/',
    'StanfordCars': 'data/stanford_cars/cars_test/',
    'OxfordPets': 'data/oxford_pets/images/',
    'OxfordFlowers': 'data/oxford_flowers/jpg/',
    }

def load_data(dataset_name):

    path = f"FOCI-Benchmark/data/{dataset_name}.json"
    img_path = img_pth[dataset_name]

    with open(path, 'r') as f:
        data = json.load(f)
    all_data = []

    for k, v in data.items():
        all_data += v

    random.shuffle(all_data)

    data_dict = {}
    few_shot_data = {}
    ground_truth_seen = {}

    for d in all_data:
        if dataset_name == 'FGVCAircraft':
            img = img_path + d['image'].split('/')[-1]

        elif dataset_name == 'Food101':
            img = img_path + d['image']

        elif dataset_name == 'OxfordFlowers':
            img = img_path + d['image'].split('/')[-1]

        elif dataset_name == 'StanfordCars':
            img = img_path + d['image'].split('/')[-1]

        elif dataset_name == 'OxfordPets':
            img = img_path + d['image'].split('/')[-1]

        if os.path.exists(img):
            print(f"Image {img} found!")
            options = d['options']
            ground_truth = d['groundtruth']
            data_dict[img] = {
                'options': options,
                'groundtruth': ground_truth
            }

        else:
            print(f"Image {img} not found!")

        for img, data in list(data_dict.items()):  # Use list() to avoid modifying dict while iterating
            ground_truth = data['groundtruth']
            if ground_truth not in ground_truth_seen:
                few_shot_data[img] = data
                ground_truth_seen[ground_truth] = True
                del data_dict[img]

    return data_dict, few_shot_data


def turn_data_to_vqa(base_prompt = 'What is this?', options=None, gt=None):

    letter_option = ['A. ', 'B. ', 'C. ', 'D. ']
    letter_option_ = ['A', 'B', 'C', 'D']

    random.shuffle(options)
    
    # Create the multiple-choice answers
    letter_option_multi_choice = [letter_option[i] + options[i] for i in range(len(options))]

    # Find the index of the ground truth in the shuffled options
    gt_index = options.index(gt)
    
    # Get the corresponding letter for the ground truth
    gt_letter = letter_option_[gt_index]


    extended_instruction = (base_prompt + '\n' + '\n'.join(letter_option_multi_choice)
                                    + '\n' + "Answer with the option's letter from the given choices directly.")
    
    return extended_instruction, gt_letter





