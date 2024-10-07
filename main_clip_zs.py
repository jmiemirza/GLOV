import argparse
import os 
import torch
import datetime
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
import trainers.zero_shot_gpt as models
import dataset.oxford_flowers
import dataset.fgvc_aircraft
import dataset.dtd
import dataset.eurosat
import dataset.food101
import dataset.sun397
import dataset.ucf101
import dataset.imagenet_r
import dataset.imagenet
import dataset.imagenet_s
import dataset.stanford_cars
import trainers.zero_shot_gpt
import dataset.imagenetv2
import dataset.cifar_local
import dataset.cubs
import dataset.resisc
import dataset.kinetics400
import dataset.caltech101
import dataset.imagenet_a
import dataset.places365
import clip.clip as clip
from utils.utils import *
import os
import utils.opt_utils as opt_utils
import pandas as pd
import time 
from utils.llm_opt_helpers import *
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import time 

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.txt_cls = args.txt_cls
    cfg.gpt_prompts = args.gpt_prompts
    cfg.text_emb = args.text_emb
    cfg.mixtral = args.mixtral
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.alpha = args.alpha

    add_waffle_opts(cfg)

def setup_cfg(args):



    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    if args.corruption:
        cfg.level = 3

    return cfg


class lossmeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


@torch.no_grad()
def zero_shot(model, dataset_name, cfg, test_loader):
    print('-------------- ZERO SHOT INFERENCE --------------')
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

                out = model(img)

                logits_base = out
                pred_base = torch.argmax(logits_base, dim=1)
                for j in range(len(label)):
                    total += 1.
                    if pred_base[j] == label[j]:
                        correct_base += 1.
            top1 = (correct_base / total) * 100
            print(f"Top-1 accuracy standard: {top1:.2f}")

            import pandas as pd

            base = f'results/{args.text_emb}'

            Path(base).mkdir(exist_ok=True, parents=True)

            result_file = f'{base}/{args.text_emb}_{backbone}.csv'

            try:
                existing_results = pd.read_csv(result_file)
            except FileNotFoundError:
                existing_results = pd.DataFrame(columns=[f'Dataset', 'Accuracy'])

            accuracy = top1
            result_entry = pd.DataFrame({'Dataset': [dataset_name], 'Accuracy': [accuracy]})
            existing_results = pd.concat([existing_results, result_entry], ignore_index=True)
            existing_results.to_csv(result_file, index=False)

            return

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

    img_emb = features['features']
    gt = features['targets']

    out = model(img_emb.cuda())

    pred_base = torch.argmax(out, dim=1)

    for j in range(len(gt)):
        total += 1.
        if pred_base[j] == gt[j]:
            correct_base += 1.

    top1 = (correct_base / total) * 100

    results_runs.append(top1)

    top1 = np.mean(results_runs)

    import pandas as pd

    base = f'results/{args.text_emb}'

    Path(base).mkdir(exist_ok=True, parents=True)

    result_file = f'{base}/{args.text_emb}_{backbone}.csv'

    try:
        existing_results = pd.read_csv(result_file)
    except FileNotFoundError:
        existing_results = pd.DataFrame(columns=[f'Dataset', 'Accuracy'])

    accuracy = top1
    result_entry = pd.DataFrame({'Dataset': [dataset_name], 'Accuracy': [accuracy]})
    existing_results = pd.concat([existing_results, result_entry], ignore_index=True)
    existing_results.to_csv(result_file, index=False)
    print(f"Top-1 accuracy standard: {top1:.2f}")

        
def main(args):
    cfg = setup_cfg(args)
    backbone = cfg.MODEL.BACKBONE.NAME.replace('/', '_')
    args.arch = backbone
    dataset_name = cfg.DATASET.NAME

    if dataset_name == 'kinetics400':
        args.batch_size = args.batch_size
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed
    cfg.mix_gpt_with_temp = args.mix_gpt_with_temp # many of these are for ablations only
    cfg.mix_mixtral_with_temp = args.mix_mixtral_with_temp
    cfg.mix_gpt_with_mixtral = args.mix_gpt_with_mixtral
    cfg.mix_all = args.mix_all
    cfg.mix_minus_cls = args.mix_minus_cls
    cfg.gpt_minus_cls = args.gpt_minus_cls
    cfg.truncate = args.truncate
    cfg.llm_type = args.type

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    model = trainer.model
    model.args = args
    test_loader = trainer.test_loader
    
    zero_shot(model, dataset_name, cfg, test_loader)


def add_waffle_opts(cfg): # for waffle -- currently disabled from this repo

    cfg.waffle_count = args.waffle_count
    cfg.randomization_budget = args.randomization_budget
    cfg.reps = args.reps
    cfg.save_model = args.save_model
    cfg.merge_predictions = args.merge_predictions
    cfg.dont_apply_descriptor_modification = args.dont_apply_descriptor_modification
    cfg.descriptor_separator = args.descriptor_separator
    cfg.pre_descriptor_text = args.pre_descriptor_text
    cfg.label_after_text = args.label_after_text
    cfg.label_before_text = args.label_before_text
    cfg.vmf_scale = args.vmf_scale
    cfg.savename = args.savename
    cfg.apply_descriptor_modification = not cfg.dont_apply_descriptor_modification

if __name__ == "__main__":
    import args as arguments
    args = arguments.get_args()
    setup_log_folder_(args)
    main(args)
