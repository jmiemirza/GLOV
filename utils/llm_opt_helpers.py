import json
import re
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
# from clip.clip import clip
import torch.nn.functional as F

def save_img_embeddings(model, loader, args, dataset_name, path):
    print('-------------- SAVING IMAGE EMBEDDINGS --------------')
    image_emb = {}
    img_emb_list = list()
    img_gt_list = list()
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            images = inputs['img']
            target = inputs['label']
            img_emb = model.image_features(images.cuda())
            img_emb_list.append(img_emb.cpu())
            img_gt_list.append(target.cpu())
    img_emb_list = torch.cat(img_emb_list, dim=0)
    img_gt_list = torch.cat(img_gt_list, dim=0)
    image_emb = {'features':img_emb_list, 'targets':img_gt_list}
    torch.save(image_emb, os.path.join(path, dataset_name + '.pth'))

@torch.no_grad()
def save_audio_embeddings(model, loader, dataset_name, path):
    print('-------------- SAVING AUDIO EMBEDDINGS --------------')
    image_emb = {}
    img_emb_list = list()
    img_gt_list = list()
    
    for batch_number, batch in enumerate(tqdm(loader)):        
        img_emb, target = load_and_transform_audio_data(batch["impath"], 'cuda', batch["label"], model)
        img_emb_list.append(img_emb.cpu())
        img_gt_list.append(target.cpu())

    img_emb_list = torch.cat(img_emb_list, dim=0)
    img_gt_list = torch.cat(img_gt_list, dim=0)
    image_emb = {'features':img_emb_list, 'targets':img_gt_list}
    torch.save(image_emb, os.path.join(path, dataset_name + '.pth'))
        # out = model(out_features)



@torch.no_grad()
def load_and_transform_audio_data(
    audio_paths,
    device,
    gt_audio,
    model, 
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    gt_idx_final = list()


    for ii, (i, audio_path) in (enumerate(zip(gt_audio, audio_paths))):
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

            gt_idx_final.append(i)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]
        all_clips = torch.stack(all_clips, dim=0)
        all_clips = {ModalityType.AUDIO: all_clips}
        all_clips = model.encode_audio(all_clips)
        audio_outputs.append(all_clips)

    return torch.cat(audio_outputs, dim=0), torch.tensor(gt_idx_final)


def dump_json_responses(data, path, i=None):

    with open(f'{path}/global_responses.json', 'w') as f:
                json.dump(data, f, indent=4)

def parse_json_respones(data, i=None, args=None): # should return a list of prompts to be used in the next step
    cleaned_prompts = []
    final_matched_prompts = list()

    if 'gpt' in args.llm:
        prompts = data['choices'][0]['message']['content']
        pattern = re.compile(r'^\d+\.\s')
        pattern_1 = re.compile(r'^\d+\:\s')
        after_period_pattern = re.compile(r'\..*')
        lines = prompts.split('\n')

        for line in lines:
            line = line.strip()
            match_prompt = pattern.search(line)
            match_prompt_1 = pattern_1.search(line)
            if match_prompt:
                cleaned= pattern.sub('', line)
                cleaned = cleaned.replace('Prompt: ', '')
                cleaned = after_period_pattern.sub('.', cleaned)    
                cleaned_prompts.append(cleaned)
                final_matched_prompts.append(cleaned.replace('<category>', '{}'))

            elif match_prompt_1:
                cleaned= pattern_1.sub('', line)
                cleaned = cleaned.replace('Prompt: ', '')
                cleaned = after_period_pattern.sub('.', cleaned)    
                cleaned_prompts.append(cleaned)
                final_matched_prompts.append(cleaned.replace('<category>', '{}'))
            else:   
                pass
    else: 
        prompts = data['choices'][0]['message']['content']

        lines = prompts.split('\n')

        for line in lines:
            # line = line.replace('Here\'s a new prompt template:','')
            # line = line.replace('Here\\\'s a new prompt template:','')
            # line = line.replace('Here\\\'s a new prompt template: ','')
            # line = line.replace('Here is a new prompt template: ','')
            # line = line.replace('Here is a new prompt template:','')
            # line = line.replace('"','')
            line = line.split(':')[-1]
            # line = line.replace('\"', '')
            cleaned_prompts.append(line)


            final_matched_prompts.append(line.replace('<category>', '{}'))


    print(final_matched_prompts)

    return final_matched_prompts, cleaned_prompts

def evaluate_prompts(prompts, model, cfg, loader, dataset_name): # should return the accuracy of the model with the given prompts

    print('-------------- EVALUATING PROMPTS --------------')

    assert isinstance(prompts, list)
    backbone = cfg.MODEL.BACKBONE.NAME

    backbone = backbone.replace('/', '_')
    few_shot_embeddings_path = f'saved_embeddings/few_shot_embeddings/shots_{cfg.DATASET.NUM_SHOTS}/{backbone}'
    Path(few_shot_embeddings_path).mkdir(exist_ok=True, parents=True)

    if Path(f'{few_shot_embeddings_path}/{dataset_name}.pth').exists():
        features = torch.load(f'{few_shot_embeddings_path}/{dataset_name}.pth')

    else:

        path_to_save = f'{few_shot_embeddings_path}'
        save_img_embeddings(model, loader, cfg, dataset_name, path_to_save)
        features = torch.load(f'{few_shot_embeddings_path}/{dataset_name}.pth')

    img_emb = features['features'].cuda()
    gt = features['targets']

    accuracies = list()

    correct_base = 0.
    total = 0.

    mistakes_idx_global = list()
    for prompt in prompts:
        mistakes_idx = list()
        prompt = prompt.replace('<category>', '{}') # just in case it's not replaced in the previous step
        text_features = model.txt_features(model.classes, [prompt])
        text_features = text_features.cuda()
        out = img_emb.float() @ text_features.float()

        pred_base = torch.argmax(out, dim=1)
        for j in range(len(gt)):
            total += 1.
            if pred_base[j] == gt[j]:
                correct_base += 1.
            else: 
                gt_idx = gt[j].item()
                mistakes_idx.append(gt_idx)


        top1 = (correct_base / total) * 100
        accuracies.append(top1)
        total = 0.
        correct_base = 0.

        mistakes_idx_global.append(mistakes_idx)
    return np.array(accuracies), mistakes_idx_global


def get_previous_prompts(df):
    prompts = list()
    for prompt in df['Prompts'].values:


        prompts+=prompt

    accuracies = list()
    for acc in df['Accuracy'].values:
        accuracies+=acc

    sort = np.argsort(accuracies)
    sorted_prompts = np.array(prompts)[sort]
    sorted_accuracies = np.array(accuracies)[sort]
    
    print(f'Prompts: {sorted_prompts}')
    print(f'Accuracies: {sorted_accuracies}')

    return sorted_prompts, sorted_accuracies

def get_prompts_from_json(path):
    with open(path+'/global_responses.json', 'r') as f:
        data = json.load(f)

    prompts = list()
    for i in range(len(data)):
        for key, value in data[i].items():
            if value['gpt_prompts_sorted'] == []:
                continue
            prompts.append(value['gpt_prompts_sorted'][-1])
    return prompts

def get_last_prompts(path):
    with open(path+'/global_responses.json', 'r') as f:
        data = json.load(f)

    prompts = list()

    key, value = data[-1].items()
    prompts = value['gpt_prompts_sorted']
    return prompts

def get_prompts_from_json_and_acc(path):
    with open(path+'/global_responses.json', 'r') as f:
        data = json.load(f)

    prompts = list()
    accuracy = list()
    for i in range(len(data)):
        for key, value in data[i].items():
            if value['gpt_prompts_sorted'] == []:
                continue
            prompts+=value['gpt_prompts_sorted']
            accuracy+=value['gpt_prompts_accuracies_sorted']
    sorted_acc_idx = np.argsort(accuracy)
    sorted_prompts = np.array(prompts)[sorted_acc_idx]

    return sorted_prompts.tolist()


def eval_proxy_tuning(prompts, test_loader, model, cfg, dataset_name, clip=None, n=None):
    # step = step if step is not None else 'final'        
    print(f'-------------- Evaluating Prompts Found Through Proxy Tuning --------------')
    prompts_b = prompts[-n:]
    text_features = model.txt_features(model.classes, prompts_b)
    total = 0.
    correct_base = 0.
    model.eval()

    backbone = cfg.MODEL.BACKBONE.NAME

    backbone = backbone.replace('/', '_')


    if dataset_name == 'kinetics400': # for kinetics400, for some reason couldn't save it through the other way (always ran out of mem)
        import pickle
        precomputed_encs_folder = 'saved_embeddings/kinetics400'

        Path(precomputed_encs_folder).mkdir(exist_ok=True, parents=True)

        backbone_ = backbone.replace('_', '')
        precomputed_encs_file = os.path.join(
            precomputed_encs_folder,
            f'k400_{backbone_.lower().replace("/", "")}.pkl'
        )


        if not os.path.exists(precomputed_encs_file):
            print('-------------- NO IMAGE EMBEDDINGS FOUND --------------')
            print('Saving Images Embeddings...')

            enc_coll = []
            label_coll = []
            with torch.no_grad():
                for batch_number, batch in enumerate(tqdm(test_loader, desc='Precomputing image embeddings...')):
                    images = batch["img"]
                    labels = batch["label"]

                    images = images.cuda()
                    labels = labels.cuda()
                    label_coll.append(labels)

                    image_encodings = F.normalize(model.image_features(images))
                    enc_coll.append(image_encodings.cpu())
            load_res = {'enc': enc_coll, 'labels': label_coll}
            pickle.dump(load_res, open(precomputed_encs_file, 'wb'))



        load_res = pickle.load(open(precomputed_encs_file, 'rb'))

        img_emb = load_res['enc']
        labels = load_res['labels']


        with torch.no_grad():
            for i, (img, label) in enumerate(tqdm(zip(img_emb, labels), total=len(img_emb))):

                img = img.cuda()
                label = label.cuda()

                text_features = text_features.cuda()
                out = img_emb.float() @ text_features.float()

                logits_base = out
                pred_base = torch.argmax(logits_base, dim=1)
                for j in range(len(label)):
                    total += 1.
                    if pred_base[j] == label[j]:
                        correct_base += 1.
            top1 = (correct_base / total) * 100
            print(f"Top-1 accuracy standard: {top1:.2f}")

    if cfg.MODEL.BACKBONE.NAME in clip._MODELS: # for simple clip
            backbone = backbone.replace('/', '_')
            base_path = f'saved_embeddings/all_img_emb_{backbone}/'

    elif 'quickgelu' in cfg.MODEL.BACKBONE.NAME:
        backbone = backbone.replace('/', '_')
        base_path = f'saved_embeddings/all_img_emb_meta_clip_{backbone}/' # long names can be improved, but whatever

    elif 'image_bind' in cfg.MODEL.BACKBONE.NAME:	
        backbone = backbone.replace('/', '_')	
        base_path = f'saved_embeddings/all_img_emb_image_bind_{backbone}/'
    else:
        raise NotImplementedError

    if Path(f'{base_path}/{dataset_name}.pth').exists():
        features = torch.load(f'{base_path}/{dataset_name}.pth')
    else:
        if backbone == 'image_bind':
            save_audio_embeddings(model, test_loader, dataset_name, base_path)
            features = torch.load(f'{base_path}/{dataset_name}.pth')
        else:
            Path(base_path).mkdir(exist_ok=True, parents=True)
            print('-------------- NO IMAGE EMBEDDINGS FOUND --------------')
            save_img_embeddings(model, test_loader, cfg, dataset_name, base_path)
            features = torch.load(f'{base_path}/{dataset_name}.pth')

    img_emb = features['features'].cuda()
    gt = features['targets'].cuda()
    text_features = text_features.cuda()
    out = img_emb.float() @ text_features.float()
    pred_base = torch.argmax(out, dim=1)
    for j in range(len(gt)):
        total += 1.
        if pred_base[j] == gt[j]:
            correct_base += 1.
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy on test set: {top1:.2f}")

def evaluate_best_of_best(path, test_loader, model, cfg, dataset_name, step=None, clip=None):
    # step = step if step is not None else 'final'        
    print(f'-------------- Evaluating Best Prompts step ::: {step}--------------')
    prompts = get_prompts_from_json_and_acc(path) # returns a list of best prompts
    prompts_w = prompts[:10]


    best_p_words = list()
    worst_p_words = list()

    prompts_b = prompts[-100:]

    print(len(prompts_b))

    text_features = model.txt_features(model.classes, prompts_b)

    total = 0.
    correct_base = 0.
    model.eval()

    backbone = cfg.MODEL.BACKBONE.NAME

    backbone = backbone.replace('/', '_')


    if dataset_name == 'kinetics400': # for kinetics400, for some reason couldn't save it through the other way (always ran out of mem)
        import pickle
        precomputed_encs_folder = 'saved_embeddings/kinetics400'

        Path(precomputed_encs_folder).mkdir(exist_ok=True, parents=True)

        backbone_ = backbone.replace('_', '')
        precomputed_encs_file = os.path.join(
            precomputed_encs_folder,
            f'k400_{backbone_.lower().replace("/", "")}.pkl'
        )


        if not os.path.exists(precomputed_encs_file):
            print('-------------- NO IMAGE EMBEDDINGS FOUND --------------')
            print('Saving Images Embeddings...')

            enc_coll = []
            label_coll = []
            with torch.no_grad():
                for batch_number, batch in enumerate(tqdm(test_loader, desc='Precomputing image embeddings...')):
                    images = batch["img"]
                    labels = batch["label"]

                    images = images.cuda()
                    labels = labels.cuda()
                    label_coll.append(labels)

                    image_encodings = F.normalize(model.image_features(images))
                    enc_coll.append(image_encodings.cpu())
            load_res = {'enc': enc_coll, 'labels': label_coll}
            pickle.dump(load_res, open(precomputed_encs_file, 'wb'))



        load_res = pickle.load(open(precomputed_encs_file, 'rb'))

        img_emb = load_res['enc']
        labels = load_res['labels']


        with torch.no_grad():
            for i, (img, label) in enumerate(tqdm(zip(img_emb, labels), total=len(img_emb))):

                img = img.cuda()
                label = label.cuda()

                text_features = text_features.cuda()
                out = img_emb.float() @ text_features.float()

                logits_base = out
                pred_base = torch.argmax(logits_base, dim=1)
                for j in range(len(label)):
                    total += 1.
                    if pred_base[j] == label[j]:
                        correct_base += 1.
            top1 = (correct_base / total) * 100
            print(f"Top-1 accuracy standard: {top1:.2f}")

            import pandas as pd

            Path(path).mkdir(exist_ok=True, parents=True)

            result_file = f'{path}/{backbone}_results.csv'

            try:
                existing_results = pd.read_csv(result_file)
            except FileNotFoundError:
                existing_results = pd.DataFrame(columns=['Step', 'Dataset', 'Accuracy', 'Path', 'Prompts', 'Few Shot'])

            accuracy = top1
            result_entry = pd.DataFrame({'Step': [step], 'Dataset': [dataset_name], 'Accuracy': [accuracy], 'Path': [path], 'Prompts' : [prompts]})
            existing_results = pd.concat([existing_results, result_entry], ignore_index=True)
            existing_results.to_csv(result_file, index=False)

            return accuracy
        

    

    if cfg.MODEL.BACKBONE.NAME in clip._MODELS: # for simple clip
        backbone = backbone.replace('/', '_')
        base_path = f'saved_embeddings/all_img_emb_{backbone}/'

    elif 'quickgelu' in cfg.MODEL.BACKBONE.NAME:
        backbone = backbone.replace('/', '_')
        base_path = f'saved_embeddings/all_img_emb_meta_clip_{backbone}/' # long names can be improved, but whatever

    elif 'image_bind' in cfg.MODEL.BACKBONE.NAME:	
        backbone = backbone.replace('/', '_')	
        base_path = f'saved_embeddings/all_img_emb_image_bind_{backbone}/'
    else:
        raise NotImplementedError

    if Path(f'{base_path}/{dataset_name}.pth').exists():
        features = torch.load(f'{base_path}/{dataset_name}.pth')
    else:
        if backbone == 'image_bind':
            save_audio_embeddings(model, test_loader, dataset_name, base_path)
            features = torch.load(f'{base_path}/{dataset_name}.pth')
        else:
            Path(base_path).mkdir(exist_ok=True, parents=True)
            print('-------------- NO IMAGE EMBEDDINGS FOUND --------------')
            save_img_embeddings(model, test_loader, cfg, dataset_name, base_path)
            features = torch.load(f'{base_path}/{dataset_name}.pth')

    results_runs = list()

    img_emb = features['features'].cuda()
    gt = features['targets'].cuda()

    text_features = text_features.cuda()
    out = img_emb.float() @ text_features.float()

    pred_base = torch.argmax(out, dim=1)

    for j in range(len(gt)):
        total += 1.
        if pred_base[j] == gt[j]:
            correct_base += 1.

    top1 = (correct_base / total) * 100

    results_runs.append(top1)

    top1 = np.mean(results_runs)

    import pandas as pd

    # base = f'results/{args.text_emb}'

    Path(path).mkdir(exist_ok=True, parents=True)

    result_file = f'{path}/{backbone}_results.csv'

    try:
        existing_results = pd.read_csv(result_file)
    except FileNotFoundError:
        existing_results = pd.DataFrame(columns=['Step', 'Dataset', 'Accuracy', 'Path', 'Prompts', 'Few Shot'])

    accuracy = top1
    result_entry = pd.DataFrame({'Step': [step], 'Dataset': [dataset_name], 'Accuracy': [accuracy], 'Path': [path], 'Prompts' : [prompts]})
    existing_results = pd.concat([existing_results, result_entry], ignore_index=True)
    existing_results.to_csv(result_file, index=False)

    
    print(f"Top-1 accuracy on test set: {top1:.2f}")

def evaluate_best_prompts(prompts, test_loader, model, cfg, dataset_name, step=None, clip=None, last=False, few_shot=False, few_shot_embeddings_path=None, val=False, val_path=None):
    backbone = cfg.MODEL.BACKBONE.NAME
    backbone = backbone.replace('/', '_')
    
    step = step if step is not None else 'final'        

    if few_shot:
        base_path = f'saved_embeddings/train/shots_{cfg.DATASET.NUM_SHOTS}_{cfg.SEED}/{backbone}'

    elif val: 
        base_path = f'saved_embeddings/val/shots_{cfg.DATASET.NUM_SHOTS}_{cfg.SEED}/{backbone}'

    else:
        base_path = f'saved_embeddings/test/shots_{cfg.DATASET.NUM_SHOTS}_{cfg.SEED}/{backbone}'
    
    
    print(f'-------------- Evaluating Best Prompts step ::: {step}--------------')

    if step  == 0:
        valid_propmpts = ['A photo of a {}.']
    else:
        # prompts = get_last_prompts(path) # returns a list of best prompts
        
        len_p = len(prompts)

        if len_p < 3:
            n = len_p

        else: 
            n = 3
        
        if not last:
            prompts = prompts[-n:]

            valid_propmpts = list()
            for template in prompts:
                empty_braces_pattern = re.compile(r'\{\s*\}')

                if not empty_braces_pattern.search(template):
                    continue
                valid_propmpts.append(template)
    text_features = model.txt_features(model.classes, valid_propmpts)

    total = 0.
    correct_base = 0.
    model.eval()

    if Path(f'{base_path}/{dataset_name}.pth').exists():
        features = torch.load(f'{base_path}/{dataset_name}.pth')
    else:
        if backbone == 'image_bind':
            save_audio_embeddings(model, test_loader, dataset_name, base_path)
            features = torch.load(f'{base_path}/{dataset_name}.pth')
        else:
            Path(base_path).mkdir(exist_ok=True, parents=True)
            print('-------------- NO IMAGE EMBEDDINGS FOUND --------------')
            save_img_embeddings(model, test_loader, cfg, dataset_name, base_path)
            features = torch.load(f'{base_path}/{dataset_name}.pth')

    results_runs = list()

    img_emb = features['features'].cuda()
    gt = features['targets'].cuda()

    text_features = text_features.cuda()
    out = img_emb.float() @ text_features.float()

    pred_base = torch.argmax(out, dim=1)

    for j in range(len(gt)):
        total += 1.
        if pred_base[j] == gt[j]:
            correct_base += 1.

    top1 = (correct_base / total) * 100
    
    if few_shot:
        print(f"Top-1 accuracy train: {top1:.2f}")

    elif val:
        print(f"Top-1 accuracy val: {top1:.2f}")
    else:
        print(f"Top-1 accuracy on test set: {top1:.2f}")

    return top1, valid_propmpts





def eval_prompts(prompts, model, cfg, dataset_name, step=None):

    def get_individual_acc(features, text_features):
        total = 0.
        correct_base = 0.
        model.eval()
        img_emb = features['features'].cuda()
        gt = features['targets'].cuda()

        text_features = text_features.cuda()
        out = img_emb.float() @ text_features.float()

        pred_base = torch.argmax(out, dim=1)

        for j in range(len(gt)):
            total += 1.
            if pred_base[j] == gt[j]:
                correct_base += 1.

        return (correct_base / total) * 100

    backbone = cfg.MODEL.BACKBONE.NAME
    backbone = backbone.replace('/', '_')
    
    step = step if step is not None else 'final'        

    train_path = f'saved_embeddings/train/shots_{cfg.DATASET.NUM_SHOTS}_{cfg.SEED}/{backbone}'

    val_path = f'saved_embeddings/val/shots_{cfg.DATASET.NUM_SHOTS}_{cfg.SEED}/{backbone}'

    test_path = f'saved_embeddings/test/shots_{cfg.DATASET.NUM_SHOTS}_{cfg.SEED}/{backbone}'
    
    
    print(f'-------------- Evaluating Best Prompts step ::: {step}--------------')

    if step  == 0:
        prompts = ['A photo of a {}.']
    else:
        prompts = prompts # returns a list of best prompts
    text_features = model.txt_features(model.classes, prompts)



    train_features = torch.load(f'{train_path}/{dataset_name}.pth')
    test_features = torch.load(f'{test_path}/{dataset_name}.pth')
    val_features = torch.load(f'{val_path}/{dataset_name}.pth')

    return get_individual_acc(train_features, text_features), get_individual_acc(test_features, text_features), get_individual_acc(val_features, text_features)



def eval_prompts_llm_bb(prompts, model, cfg, dataset_name):

    def get_individual_acc(features, text_features):
        total = 0.
        correct_base = 0.
        model.eval()
        img_emb = features['features'].cuda()
        gt = features['targets'].cuda()

        text_features = text_features.cuda()
        out = img_emb.float() @ text_features.float()

        pred_base = torch.argmax(out, dim=1)

        for j in range(len(gt)):
            total += 1.
            if pred_base[j] == gt[j]:
                correct_base += 1.

        return (correct_base / total) * 100

    backbone = cfg.MODEL.BACKBONE.NAME
    backbone = backbone.replace('/', '_')
    test_path = f'saved_embeddings/test/shots_{cfg.DATASET.NUM_SHOTS}_{cfg.SEED}/{backbone}'
    text_features = model.txt_features(model.classes, prompts)
    test_features = torch.load(f'{test_path}/{dataset_name}.pth')
    return get_individual_acc(test_features, text_features)



def update_df(df_logs, i, prompt, accuracy, meta_prompt='none', raw_responses='none', failed=False, proxy_tuning=False, alpha=None):
    new_row = pd.DataFrame([{'Steps': i, 'Prompts': prompt, 'Accuracy': accuracy, 'Meta Prompt': meta_prompt, 
                             'Raw Responses': raw_responses, 'Failed': failed, 'Proxy Tuning': proxy_tuning, 'Alpha': alpha}])
    df_logs = pd.concat([df_logs, new_row], ignore_index=True)
    return df_logs