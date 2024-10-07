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
from utils.discovered_prompts import *

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

class CLIP_Zero_Shot_adapt(nn.Module):

    def __init__(self, model, classes, templates, device='cuda', dataset_name=None, log=None, txt_cls = None, cfg=None):
        super(CLIP_Zero_Shot_adapt, self).__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.device = device
        self.classes = classes
        self.model = model.to(device)
        self.log = log
        self.args = None
        self.txt_cls = txt_cls
        self.templates = templates

        if 'quickgelu' in self.cfg.MODEL.BACKBONE.NAME: # for meta_clip
            self.tokenizer = open_clip

        elif cfg.MODEL.BACKBONE.NAME in clip._MODELS:
            self.tokenizer = clip
        
        elif 'image_bind' in self.cfg.MODEL.BACKBONE.NAME:
            pass

        else:
            raise ValueError(f'Backbone {self.cfg.MODEL.BACKBONE.NAME} not supported')

        if cfg.text_emb == 's_temp':
            self.templates = ['a photo of a {}.']
            self.text_embeddings_for_zero_shot = self.txt_features(self.classes, self.templates)

        elif cfg.text_emb == 'glov_wo_guidance':
            self.templates = lov_templates_clip[dataset_name]
            self.text_embeddings_for_zero_shot = self.txt_features(self.classes, self.templates)

        elif cfg.text_emb == 'glov':
            self.templates = lov_guidance_templates_clip[dataset_name]
            self.text_embeddings_for_zero_shot = self.txt_features(self.classes, self.templates)

        else:
            raise NotImplementedError
        

    def image_features(self, images):
        with torch.no_grad():
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
class clip_adapt(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")


        clip_model = load_clip_to_cpu(cfg)



        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")
        self.model = CLIP_Zero_Shot_adapt(model=clip_model, classes=classnames,
                                          templates=CUSTOM_TEMPLATES[cfg.DATASET.NAME], dataset_name = cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg)
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """


        if 'image_bind' not in self.cfg.MODEL.BACKBONE.NAME:
            te_transform = te_transform_clip

        elif 'image_bind' in self.cfg.MODEL.BACKBONE.NAME and self.cfg == 'esc':
            return  
        
        else:
            te_transform = data_transform_image_bind

        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=tr_transforms)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

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
