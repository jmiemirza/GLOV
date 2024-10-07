import argparse
import os 
import torch
import datetime
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
import trainers.llava_trainer as models
import dataset.oxford_flowers
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
import trainers.llava_trainer
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
import utils.opt_utils_llava as opt_utils_llava
import pandas as pd
import time 
from utils.llm_opt_helpers_llava import *
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import login
login(token="")
import utils.llava_eval_functions as llava_eval_functionss
import time 
def initialize_writer(args, run_name):
    time_date_stamp = time.strftime("%Y%m%d")
    log_dir = f"llava_experiments_logs/metrics/{time_date_stamp}/all_tokens_diff/{args.exp_name}_{run_name}"  # Specify your log directory
    writer = SummaryWriter(log_dir)
    return writer

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


def llm_opt(model, dataset_name, cfg, test_loader, few_shot_loader, clip=None, 
            args=None, universal_path_responses=None, class_names=None, val_loader=None):

    print('-------------- OPTIMIZATION TASK --------------')
    backbone = cfg.MODEL.BACKBONE.NAME.replace('/', '_')
    if 'llama' in args.llm:
        from trainers.llama_model_llava import LlamaModel
        llm_model = LlamaModel(
            base_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct", 
                               alpha=args.alpha, args=args
                               )
        print('Loaded LLAMA model...')
    else:
        llm_model = 'GPT'

    # initialize a dataframe for the responses
    global_accuracies = None
    global_prompts = None
    df_logs = pd.DataFrame(columns=['Steps', 'Prompts', 'Accuracy', 'Meta Prompt', 'Raw Responses', 'Failed'])
    global_steps = args.global_steps # stopping criteria
    global_log = list()
    best_global_accuracy_list = list()

    run_name = f'alpha_{args.alpha}_seed_{args.seed}_layer_{args.diff_layer}_ema_{args.ema_alpha}_shots_{args.num_shots}_{backbone}_{dataset_name}'
    writer = initialize_writer(args, run_name)

    prompts_dict = dict()
    for i in range(global_steps):
        print(f'Optimization step {i + 1}/{global_steps}')

        results = dict()
        if i == 0:
            prompt = 'Describe the category present in this image briefly and also identify the name of the category present.'
            accuracy = evaluate_prompts([prompt], model, cfg, few_shot_loader, dataset_name=dataset_name) # works good!!!

            base_best_accuracy = accuracy.max()

            df_logs = update_df(df_logs, i, [prompt], accuracy.tolist(), failed=False, proxy_tuning=False)
            writer.add_scalar(f'accuracy_train', base_best_accuracy, i)
            results['accuracies'] = base_best_accuracy
            results['prompts'] = [prompt]
            prompts_dict[f'{i}_{prompt}'] = results

        if 'llama' in args.llm:

            if args.mistakes_context == 'False':
                response, meta_prompt, task_info, logits = opt_utils_llava.main(dataset_name=dataset_name, 
                                                step=i, accuracy=accuracy, dataset_info=dataset_info,
                                                response_dict=global_log, global_accuracies=global_accuracies,
                                                global_responses=global_prompts, df_logs=df_logs, 
                                                top_bottom_k=args.top_bottom, prompt='Describe the category present in this image briefly and also identify the name of the category present.', 
                                                args=args, llm_model=llm_model, writer = writer)


            response['prompt'] = prompt
            response['meta_prompt'] = meta_prompt
            prompt, prompts_for_next_step = parse_json_respones(response, i, args=args) # get the prompts for the next step after parsing
            prompt = list(filter(None, prompt))
            prompts_for_next_step = list(filter(None, prompts_for_next_step))

            if len(prompt) == 0 or len(prompt) < 2:
                failed_flag = True
                count  = 0 # just to make sure it doesn't go into an infinite loop
            else:
                failed_flag = False

            accuracy_per_prompt = evaluate_prompts(prompt, model, cfg, few_shot_loader, dataset_name=dataset_name) # returns a list of accuracies for each prompt
            
            sorted_idx = np.argsort(accuracy_per_prompt)

            sorted_prompts = np.array(prompt)[sorted_idx]
            sorted_accuracies = accuracy_per_prompt[sorted_idx]

            response['gpt_prompts_sorted'] = sorted_prompts.tolist()
            response['gpt_prompts_accuracies_sorted'] = sorted_accuracies.tolist() 
            global_log.append({f'step_{i}': response})
            dump_json_responses(global_log, universal_path_responses, i) # save the current responses

            df_logs = update_df(df_logs, i, prompts_for_next_step, accuracy_per_prompt.tolist(), 
                                meta_prompt=meta_prompt, raw_responses=response, failed=failed_flag, proxy_tuning=args.do_proxy_tuning,
                                alpha=args.alpha) # update the dataframe with the new prompts and accuracies
            
            global_prompts, global_accuracies = get_previous_prompts(df_logs)

            best_global_accuracy_list.append(global_accuracies.max())
            results['accuracies'] = sorted_accuracies
            for_logging = sorted_accuracies[-2:]
            writer.add_scalar(f'accuracy_train', sum(for_logging)/len(for_logging), i)
            results['prompts'] = sorted_prompts
            prompts_dict[f'{i}_{prompt}'] = results

            if global_accuracies[-1] > base_best_accuracy: # basically always do proxy tuning as soon as this condition is fullfiled!!!!

                print('--- Updating best accuracy ---')
                best_accuracy_idx = np.argmax(global_accuracies)
                best_acc = global_accuracies[best_accuracy_idx]
                base_best_accuracy = best_acc
                llm_model.best_prompt = global_prompts[best_accuracy_idx]
                
                second_best_idx = [i for i, x in enumerate(global_accuracies) if x < best_acc][-1]
                llm_model.worst_prompt = global_prompts[second_best_idx]

                print(f'Best prompt: {llm_model.best_prompt} --- Accuracy: {best_acc}')
                print(f'Worst prompt: {llm_model.worst_prompt} --- Accuracy: {global_accuracies[second_best_idx]}')
                args.do_proxy_tuning = True # just a hack currentl

            df_logs.to_csv(f'{args.base_path}/logs.csv', index=False)

    templates_by_iter = prompts_dict.items()
    templates_by_iter_excel_dir = os.path.join(universal_path_responses, 'results_by_iteration.csv')
    from utils.utils import write_templates_to_csv_llava
    write_templates_to_csv_llava(templates_by_iter, templates_by_iter_excel_dir)

        
def main(args):
    cfg = setup_cfg(args)
    backbone = cfg.MODEL.BACKBONE.NAME.replace('/', '_')
    args.arch = backbone
    dataset_name = cfg.DATASET.NAME

    if args.mode not in ['eval_best', 'zs']:
        universal_time_date_stamp = time.strftime("%Y%m%d-%H%M%S")
        base_path  = f'llava_experiments_logs/{args.llm}/{dataset_name}/{universal_time_date_stamp}'
        # base_path = f'results_saver/'
        Path(base_path).mkdir(exist_ok=True, parents=True)
        args.base_path = base_path
        setup_log_folder(args, base_path)
        save_args(base_path, args)
        universal_path_responses = f'{base_path}/responses'
        Path(f'{universal_path_responses}').mkdir(exist_ok=True, parents=True)

    if dataset_name == 'kinetics400':
        args.batch_size = args.batch_size
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 1
    cfg.DATALOADER.TEST.BATCH_SIZE = 1
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

    class_names = trainer.classes

    model.args = args
    
    test_loader = trainer.test_loader
    val_loader = trainer.val_loader

    train_loader = trainer.train_loader_x

    if args.mode  == 'opt':
        print('Optimization task')
        llm_opt(model, dataset_name, cfg, test_loader, few_shot_loader=train_loader, 
                clip=clip, args=args, universal_path_responses=universal_path_responses, 
                class_names = class_names, val_loader=val_loader)


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
