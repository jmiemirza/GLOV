
import argparse
# from llm_as_bb_opt.init_templates import INIT_TEMPLATES

def int_list(string):
    return list(map(int, string.split(',')))
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--label_before_text', type=str, default='A photo of a ',
                        help='Prompt-part going at the very beginning.')
    parser.add_argument('--label_after_text', type=str, default='.',
                        help='Prompt-part going at the very end.')
    ###
    parser.add_argument('--pre_descriptor_text', type=str, default='',
                        help='Text that goes right before the descriptor.')
    parser.add_argument('--descriptor_separator', type=str, default=', ',
                        help='Text separating descriptor part and classname.')
    ###
    parser.add_argument('--dont_apply_descriptor_modification', action='store_true',
                        help='Flag. If set, will not use "which (is/has/etc)" before descriptors.')
    parser.add_argument('--merge_predictions', action='store_true',
                        help='Optional flag to merge generated embeddings before computing retrieval scores.')
    parser.add_argument('--save_model', type=str, default='',
                        help='Set to a non-empty filename to store generated language embeddings & scores in a pickle file for all seed-repetitions.')
    parser.add_argument('--randomization_budget', type=int, default=15,
                        help='Budget w.r.t. to DCLIP for randomization ablations')
    parser.add_argument('--waffle_count', type=int, default=15,
                        help='For WaffleCLIP: Number of randomized descriptor pairs to use')
    parser.add_argument('--reps', type=int, default=1,
                        help='Number of repetitions to run a method for with changing randomization. Default value should be >7 for WaffleCLIP variants.')
    parser.add_argument('--savename', type=str, default='results',
                        help='Name of csv-file in which results are stored.')
    ###
    parser.add_argument('--vmf_scale', type=float, default=1)
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=7777, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--print_freq", type=int, default=10, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--exp-name', type=str, required=False)
    parser.add_argument('--dataset', type=str, default='cifar10', help="choices are ['imagenet']")
    parser.add_argument('--dataroot', default='')
    parser.add_argument('--gpt_captions_path', default='./data/gpt_prompts_cifar.json')
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--n_views', type=int, default=1)
    parser.add_argument('--arch', type=str, default='ViT-B/32', required=False)
    parser.add_argument('--inet-pretrain', type=bool, default=False)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--deep_prompt', action='store_true')
    parser.add_argument('--gpt_prompts', action='store_true')
    parser.add_argument('--text_prompts', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--entropy', action='store_true')
    parser.add_argument('--text_aug', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--txt_cls', type=str, default='17')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mixtral', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--logfolder', default='logs', type=str)
    parser.add_argument('--text_emb', default='s_temp', type=str, required=False)
    parser.add_argument('--corruption', action='store_true')
    parser.add_argument('--mix_gpt_with_temp', type=int, default=0, required=False)
    parser.add_argument('--mix_mixtral_with_temp', type=int, default=0, required=False)
    parser.add_argument('--mix_gpt_with_mixtral', type=int, default=0, required=False)
    parser.add_argument('--mix_all', type=int, default=0, required=False)
    parser.add_argument('--mix_minus_cls', type=int, default=0, required=False)
    parser.add_argument('--gpt_minus_cls', type=int, default=0, required=False)
    parser.add_argument('--truncate', type=int, default=0, required=False)
    parser.add_argument('--type', type=str, default='gpt')
    parser.add_argument('--mode', type=str, default='opt')
    parser.add_argument('--num_shots', type=int, default=1)
    parser.add_argument('--global_steps', type=int, default=100, required=False)
    parser.add_argument('--top_bottom', type=int, default=5, required=False)
    parser.add_argument('--llm', type=str, default='llama')
    parser.add_argument('--do_proxy_tuning', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=1.0, required=True)
    parser.add_argument('--normalize', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--noise', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--do_sample', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--cross_attention', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--all_layer_diff', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--mistakes_context', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--specific_layer_diff', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--mean_emb', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--cross_attention_middle_layer', type=str, default='False', required=True, choices=['True', 'False'])
    parser.add_argument('--diff_layer', type=int_list, required=True, help='List of integers separated by commas')
    parser.add_argument('--emavg', type=str, default='False', required=True)

    parser.add_argument('--temperature', type=float, default=1.0, required=True)
    parser.add_argument('--ema_alpha', type=float, default=1.0, required=True)
    parser.add_argument('--top_k', type=int, default=1, required=True)
    parser.add_argument('--top_p', type=float, default=1.0, required=True)

    parser.add_argument('--exp_name', type=str, required=True)



    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Path to the reference outputs."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/llama-3")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        default=True,
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="proxy_tuning.eval.templates.create_prompt_with_llama2_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )



    #### llm as black box optimizer paper args 

    parser.add_argument(
    "--gpt_model", type=str, default="gpt-3.5-turbo-0125", choices=["gpt-3.5-turbo-0125", "gpt-4-0314"],
    help="the model used for prompting" )
    parser.add_argument(
        "--prompt_method", type=str, default="method_1",
        help="the prompting method"
    )
    parser.add_argument(
        "--num_iters", type=int, default=100,
        help="number of iterations for prompting",
    )
    # parser.add_argument(
    #     "--temperature", type=float, default=1,
    #     help="default temperature for ChatGPT",
    # )
    # parser.add_argument(
    #     '--init_templates', type=str, default='openai', choices=INIT_TEMPLATES.keys(),
    #     help="the initial templates for prompting",
    # )
    parser.add_argument(
        '--laion_seed', type=int, default=0,
        help="seed for selecting laion templates",
    )
    parser.add_argument(
        "--template_pool_size", type=int, default=3,
        help="the size of template pool to show for ChatGPT (default: 3)",
    )
    parser.add_argument(
        "--eval", type=str, default='train', choices=['train', 'val', 'train_val', 'test'],
        help="the split used for evaluating the templates",
    )
    parser.add_argument(
        "--num_templates_from_gpt", type=int, default=1,
        help="the number of templates we asked from chatgpt",
    )
    parser.add_argument(
        "--run", type=int, default=0,
        help="the i'th run of the experiment",
    )

    ###########################
    # Directory Config (modify if using your own paths)
    ###########################
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="where the dataset is saved",
    )
    parser.add_argument(
        "--indices_dir", type=str, default="./indices", help="where the (few-shot) indices is saved",
    )
    parser.add_argument(
        "--feature_dir", type=str, default="./features", help="where to save pre-extracted features",
    )
    parser.add_argument(
        "--result_dir", type=str, default="./llm_bb_10_prompts", help="where to save experiment results",
    )
    parser.add_argument(
        "--num_prompt",
        type=int,
        default=None,
        required=True,
        help="Number of examples in few-shot context",
    )

    ###########################
    # Dataset Config (few_shot_split.py)
    ###########################
    # parser.add_argument(
    #     "--dataset", type=str, default="imagenet", choices=dataset_classes.keys(),
    #     help="dataset name",
    # )
    # parser.add_argument(
    #     "--shot", type=int, default=1, choices=[1, 2, 4, 8, 16],
    #     help="train shot number. note that val shot is automatically set to min(4, shot)",
    # )
    # parser.add_argument(
    #     "--seed", type=int, default=1, help="seed number",
    # )

    ###########################
    # Feature Extraction Config (features.py)
    ###########################
    # parser.add_argument(
    #     "--clip_encoder", type=str, default="RN50", choices=["ViT-B/16", "ViT-B/32", "RN50", "RN101"],
    #     help="specify the clip encoder to use",
    # )

    ###########################
    # LMM EVAL ARGS
    ###########################

    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )

    
    # parser.add_argument("--batch_size", type=str, default=1)
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default=None,
    #     help="Device to use (e.g. cuda, cuda:0, cpu)",
    # )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--show_task_to_terminal",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles",
    )
    # args = parser.parse_args()
    # return args




    args = parser.parse_args() # many args can be redundant, but are kept for compatibility with other scripts

    return args