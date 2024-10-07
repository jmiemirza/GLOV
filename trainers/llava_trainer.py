import os.path as osp
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
from tqdm import tqdm
from pathlib import Path
_tokenizer = _Tokenizer()
from utils.model_utils import *
from utils.templates import *
import open_clip
import numpy as np
from typing import Tuple, List, Union, Any
import csv
import re 
import numpy as np
import torch 
import transformers_local
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
from sentence_transformers import SentenceTransformer
import sys
import warnings
model_path= '/all-mpnet-base-v2/'
warnings.filterwarnings("ignore")

CUSTOM_TEMPLATES = {
    "OxfordPets": oxford_pets,
    "OxfordFlowers": oxford_flower,
    "FGVCAircraft": aircraft,
    "DescribableTextures": dtd,
    "EuroSAT": eurosat,
    "RESISC45": eurosat,
    "StanfordCars": cars,
    "Food101": food101,
    "SUN397": sun397,
    "places365": sun397,
    "Caltech101": caltech,
    "CIFAR10_local": cifar10,
    "CIFAR100_local": cifar100,
    "UCF101": ucf101,
    "kinetics400": kinetics400,
    "ImageNet": imagenet,
    "ImageNetSketch": imagenet,
    "ImageNetV2": imagenet,
    "ImageNetA": imagenet,
    "ImageNetR": imagenet,
    "ImageNetGaussian": imagenet,
    "ImageNetDefocus": imagenet,
    "CUBS200": CUBS200,
    "esc": CUBS200,
}
def read_csv_to_list(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

def has_odd_number_of_braces(template):
    return (template.count('{') % 2 != 0) or (template.count('}') % 2 != 0)

class llava_model(nn.Module):

    def __init__(self, model, classes, templates, device='cuda:0', 
                 dataset_name=None, log=None, txt_cls = None, cfg=None,
                 tokenizer= None, image_processor=None, max_length=None):
        super(llava_model, self).__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.device = device
        self.classes = classes
        self.model = model.to(device)
        self.log = log
        self.args = None
        self.txt_cls = txt_cls
        self.templates = templates
        self.tokenizer = tokenizer
        self.image_processor=image_processor
        self.max_length = max_length

        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)

        self.text_emb_classes = self.get_text_emb(self.classes)


    def get_text_emb(self, classes):

        class_embeddings = self.sentence_transformer.encode(classes)

        return class_embeddings

    def get_llava_outputs(self, x, prompt, image_sizes):
        x = [_image.to(dtype=torch.float16, device=self.device) for _image in x]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + f"\n{prompt}?"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        cont = self.model.generate(
            input_ids,
            images=x,
            image_sizes=[[image_sizes[0][0].item(), image_sizes[0][1].item()]],
            do_sample=False,
            temperature=0,
            max_new_tokens=50,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs
        
    def process_audio(self, paths, gt_audio):

        paths= [f'data/ESC-50-master/audio/{x}' for x in paths]

        assert self.dataset_name == 'esc'

        if Path('saved_embeddings/esc.pth').exists():
            data = torch.load('saved_embeddings/esc/esc.pth')
            return data
        else:
            audio_features, gt_tensor = imagebind_data_utils.load_and_transform_audio_data(paths,'cuda', gt_audio, self.model)
            inputs = {ModalityType.AUDIO: data}
            audio_features = self.model.encode_audio(inputs)

            meta_data = {'gt_audio': gt_tensor, 'files': paths, 'audio_features': audio_features}	

            torch.save(meta_data, 'saved_embeddings/esc/esc.pth')
            # print(gt_tensor.shape)
            return audio_features

    def encode_audio(self, inputs):
        audio_features = self.model.encode_audio(inputs)
        # print(audio_features.shape)
        return audio_features

    def image_features(self, images):
        with torch.no_grad():

            if 'image_bind' in self.cfg.MODEL.BACKBONE.NAME:
                    images = {ModalityType.VISION: images}
                    image_features = self.model.encode_image(images)
                    return image_features

            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features

    def txt_features(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = list()
                for template in templates:
                    empty_braces_pattern = re.compile(r'\{\s*\}')

                    if not empty_braces_pattern.search(template):
                        continue
                    
                    placeholder_count = template.count("{}")
                    if placeholder_count == 1:
                        try:
                            texts += [template.format(classname)]
                        except (ValueError, KeyError):
                            continue
                    else:
                        try:
                            filled_template = template
                            for i in range(placeholder_count):
                                filled_template = filled_template.replace("{}", classname, 1)
                            texts += [filled_template]
                        except (ValueError, KeyError):
                            continue

                texts = self.tokenize_text(texts).cuda()

                if 'image_bind' in self.cfg.MODEL.BACKBONE.NAME:
                    texts = {ModalityType.TEXT: texts}
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

        return zeroshot_weights.squeeze()
    
    def txt_features_llm_as_bb(self, classnames, templates):
        prompts = [templates.format(cls_name) for cls_name in (classnames)]
        prompts = clip.tokenize(prompts).cuda()
        text_features = self.model.encode_text(prompts)
        text_features = torch.nn.functional.normalize(text_features, dim=1)

        return text_features
    



    def tokenize_text(self, texts):

        if 'quickgelu' in self.cfg.MODEL.BACKBONE.NAME:
            texts = self.tokenizer.tokenize(texts=texts).cuda()  # tokenize

        elif self.cfg.MODEL.BACKBONE.NAME in clip._MODELS:
            texts = self.tokenizer.tokenize(texts=texts, truncate=True).cuda()  # tokenize

        elif 'image_bind' in self.cfg.MODEL.BACKBONE.NAME:
            tokenizer = SimpleTokenizer(bpe_path="ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz")
            tokens = [tokenizer(t).unsqueeze(0).cuda() for t in texts]
            texts = torch.cat(tokens, dim=0)

        return texts


    def get_embd_text(self, text):

        texts = self.tokenize_text(text)
        class_embeddings = self.model.encode_text(texts.cuda())  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()

        return class_embedding

    def zeroshot_classifier_gpt(self, classnames, mode = str):

        assert mode == 'mpvr'

        assert self.cfg.llm_type in ['gpt', 'mixtral']


        path_to_file = f'./descriptions/{self.cfg.llm_type}/{self.dataset_name}.json'

        print('Reading descriptions from ::: ', path_to_file)

        with open(path_to_file) as f:
            gpt3_prompts = json.load(f)

        if self.dataset_name == 'SUN397' or self.dataset_name == 'UCF101' or \
                self.dataset_name == 'StanfordCars_' or self.dataset_name == 'OxfordPets' or \
                self.dataset_name == 'Food101':

            classnames = gpt3_prompts.keys()

        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = []
                for t in gpt3_prompts[classname]:
                    texts.append(t)
                texts = self.tokenize_text(texts)
                class_embeddings = self.model.encode_text(texts.cuda())  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def forward(self, x1=None):
        with torch.no_grad():

            if 'esc' in self.dataset_name:
                # x1 = self.audio_emb
                out = x1.float() @ self.text_embeddings_for_zero_shot.float()
                return out
            out = x1.float() @ self.text_embeddings_for_zero_shot.float()
        return out


@TRAINER_REGISTRY.register()
class llava_adapt(TrainerX):


    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print("Building LLAVA MODEL")
        self.model = llava_model(model=self.llava, classes=classnames,
                                          templates=CUSTOM_TEMPLATES[cfg.DATASET.NAME], dataset_name = cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg, 
                                          tokenizer= self.tokenizer, image_processor=self.image_processor, max_length=self.max_length)
        self.register_model("llava_adapt", self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """

        from utils.llava_utils import CustomLlavaTransform
        import utils.llava_utils as llava_utils

        self.llava, self.tokenizer, self.image_processor, self.max_length = llava_utils.load_llava_model()
        llava_transform = CustomLlavaTransform(self.image_processor, self.llava.config) # until better solution found!!!
        
        dm = DataManager(self.cfg, custom_tfm_test=llava_transform, custom_tfm_train=llava_transform)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.classes = dm.dataset.classnames
        self.dm = dm

    def parse_batch_train(self, batch):

        if isinstance(batch, list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
            input = input.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
