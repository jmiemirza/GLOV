import torch
from tqdm import tqdm
import numpy as np

def save_accuracy_to_file(dataset, accuracy, prompt, total_samples):

    #check if csv file exists

    import os

    base_pth =  'llava_cls_results'
    if not os.path.exists(base_pth):
        os.makedirs(base_pth)

    if not os.path.exists(f'{base_pth}/accuracy.csv'):
        with open('accuracy.csv', 'w') as f:
            f.write('dataset,prompt,accuracy,datetime,total_samples\n')

    # Append the accuracy to the file
    with open(f'{base_pth}/accuracy.csv', 'a') as f:
        import datetime
        f.write(f'{dataset},{prompt},{accuracy},{datetime.datetime.now()},{total_samples}\n')

    print(f"Accuracy saved to {base_pth}/accuracy.csv")


def save_accuracy_to_file_lov(dataset, accuracy, prompt, total_samples, identifier):

    #check if csv file exists

    import os

    base_pth =  'llava_cls_results'
    if not os.path.exists(base_pth):
        os.makedirs(base_pth)

    if not os.path.exists(f'{base_pth}/accuracy_lov.csv'):
        with open('accuracy_lov.csv', 'w') as f:
            f.write('dataset,prompt,accuracy,datetime,identifier,total_samples\n')

    # Append the accuracy to the file
    with open(f'{base_pth}/accuracy_lov.csv', 'a') as f:
        import datetime
        f.write(f'{dataset},{prompt},{accuracy},{datetime.datetime.now()},{identifier},{total_samples}\n')

    print(f"Accuracy saved to {base_pth}/accuracy.csv")



def eval_zs(model, dataloader, prompts, dataset_name):


    total_samples = len(dataloader.dataset)

    for prompt in prompts:
        print('Evaluting prompt:', prompt)
        all_caption = []
        all_labels = []
        all_img_file_path = []
        for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = x['label'].cpu().numpy()  # Convert label tensor to a NumPy array on CPU
            all_labels.append(label)  # Append the NumPy array to the list

            # if i > 20:
            #     break
            
            img_file_path = x['impath']
            # print(img_file_path)

            all_img_file_path.append(img_file_path)

            img = x["img"][0].to(model.device)  # Move image to the GPU
            img_size = x["img"][1]

            # Generate captions using the model
            if isinstance(model, torch.nn.DataParallel):
                caption = model.module.get_llava_outputs(img, prompt, img_size)
            else:
                caption = model.get_llava_outputs(img, prompt, img_size)  

            all_caption += caption  # Collect all captions

            # Compute and display accuracy at each 200 time steps
            if (i + 1) % 100 == 0:
                # Generate embeddings for the captions so far
                img_emb = model.get_text_emb(all_caption)

                # Compute similarities between image embeddings and class embeddings
                similarities = model.sentence_transformer.similarity(img_emb, model.text_emb_classes)

                # Convert similarities to predictions (top-1)
                preds = similarities.argmax(dim=1).cpu().numpy()  # Get the index of the highest similarity score

                # Concatenate all labels into a single NumPy array
                all_labels_np = np.concatenate(all_labels)  # Flatten all labels

                # Calculate the accuracy
                accuracy = (preds == all_labels_np).mean()  # Compute top-1 accuracy
                print(f"Accuracy at step {i + 1}: {accuracy * 100:.2f}%")


        # write prompt, img files paths, captions to a csv file with the date and time now as the name

        import csv
        import os
        import datetime

        path = 'llava_cls_results/outputs/'
        if not os.path.exists(path):
            os.makedirs(path)

        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with open(f'llava_cls_results/outputs/{dataset_name}_{date}.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['prompt', 'img_path', 'caption'])

            for i in range(len(all_img_file_path)):
                writer.writerow([prompt, all_img_file_path[i][0], all_caption[i]])


        # Final accuracy calculation after the last step
        img_emb = model.get_text_emb(all_caption)
        similarities = model.sentence_transformer.similarity(img_emb, model.text_emb_classes)
        preds = similarities.argmax(dim=1).cpu().numpy()
        all_labels_np = np.concatenate(all_labels)
        accuracy = (preds == all_labels_np).mean()
        print(f"Final Accuracy: {accuracy * 100:.2f}%")

        save_accuracy_to_file(dataset_name, accuracy, prompt, total_samples)

    return accuracy, preds, all_caption



def eval_lov(model, dataloader, prompts, dataset_name, identifier):


    total_samples = len(dataloader.dataset)

    for prompt in prompts:
        print('Evaluting prompt:', prompt)
        all_caption = []
        all_labels = []
        all_img_file_path = []
        for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = x['label'].cpu().numpy()  # Convert label tensor to a NumPy array on CPU
            all_labels.append(label)  # Append the NumPy array to the list

            # if i > 20:
            #     break
            
            img_file_path = x['impath']
            # print(img_file_path)

            all_img_file_path.append(img_file_path)

            img = x["img"][0].to(model.device)  # Move image to the GPU
            img_size = x["img"][1]

            # Generate captions using the model
            if isinstance(model, torch.nn.DataParallel):
                caption = model.module.get_llava_outputs(img, prompt, img_size)
            else:
                caption = model.get_llava_outputs(img, prompt, img_size)  

            all_caption += caption  # Collect all captions

            # Compute and display accuracy at each 200 time steps
            if (i + 1) % 100 == 0:
                # Generate embeddings for the captions so far
                img_emb = model.get_text_emb(all_caption)

                # Compute similarities between image embeddings and class embeddings
                similarities = model.sentence_transformer.similarity(img_emb, model.text_emb_classes)

                # Convert similarities to predictions (top-1)
                preds = similarities.argmax(dim=1).cpu().numpy()  # Get the index of the highest similarity score

                # Concatenate all labels into a single NumPy array
                all_labels_np = np.concatenate(all_labels)  # Flatten all labels

                # Calculate the accuracy
                accuracy = (preds == all_labels_np).mean()  # Compute top-1 accuracy
                print(f"Accuracy at step {i + 1}: {accuracy * 100:.2f}%")


        # write prompt, img files paths, captions to a csv file with the date and time now as the name

        import csv
        import os
        import datetime

        path = 'llava_cls_results/outputs/'
        if not os.path.exists(path):
            os.makedirs(path)

        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with open(f'llava_cls_results/outputs/{dataset_name}_{date}.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['prompt', 'img_path', 'caption'])

            for i in range(len(all_img_file_path)):
                writer.writerow([prompt, all_img_file_path[i][0], all_caption[i]])


        # Final accuracy calculation after the last step
        img_emb = model.get_text_emb(all_caption)
        similarities = model.sentence_transformer.similarity(img_emb, model.text_emb_classes)
        preds = similarities.argmax(dim=1).cpu().numpy()
        all_labels_np = np.concatenate(all_labels)
        accuracy = (preds == all_labels_np).mean()
        print(f"Final Accuracy: {accuracy * 100:.2f}%")

        save_accuracy_to_file_lov(dataset_name, accuracy, prompt, total_samples, identifier)

    return accuracy, preds, all_caption


def eval_opt(model, dataloader, prompt, dataset_name):
    all_caption = []
    all_labels = []

    total_samples = len(dataloader.dataset)
    for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
        label = x['label'].cpu().numpy()  # Convert label tensor to a NumPy array on CPU
        all_labels.append(label)  # Append the NumPy array to the list

        img = x["img"][0].to(model.device)  # Move image to the GPU
        img_size = x["img"][1]

        # Generate captions using the model
        if isinstance(model, torch.nn.DataParallel):
            caption = model.module.get_llava_outputs(img, prompt, img_size)
        else:
            caption = model.get_llava_outputs(img, prompt, img_size)  

        all_caption += caption  # Collect all captions

        # Compute and display accuracy at each 200 time steps
        if (i + 1) % 100 == 0:
            # Generate embeddings for the captions so far
            img_emb = model.get_text_emb(all_caption)

            # Compute similarities between image embeddings and class embeddings
            similarities = model.sentence_transformer.similarity(img_emb, model.text_emb_classes)

            # Convert similarities to predictions (top-1)
            preds = similarities.argmax(dim=1).cpu().numpy()  # Get the index of the highest similarity score

            # Concatenate all labels into a single NumPy array
            all_labels_np = np.concatenate(all_labels)  # Flatten all labels

            # Calculate the accuracy
            accuracy = (preds == all_labels_np).mean()  # Compute top-1 accuracy
            print(f"Accuracy at step {i + 1}: {accuracy * 100:.2f}%")

    # Final accuracy calculation after the last step
    img_emb = model.get_text_emb(all_caption)
    similarities = model.sentence_transformer.similarity(img_emb, model.text_emb_classes)
    preds = similarities.argmax(dim=1).cpu().numpy()
    all_labels_np = np.concatenate(all_labels)
    accuracy = (preds == all_labels_np).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy * 100
