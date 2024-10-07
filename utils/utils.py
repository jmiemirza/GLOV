import logging
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from timm.scheduler import CosineLRScheduler
logger_initialized = {}
import getpass
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import random
from pathlib import Path
import pandas as pd



def write_templates_to_csv(templates, filename):
    def format_percent(x):
        if isinstance(x, (float)):
            return '{:.6%}'.format(x)
        else:
            return x
    columns = ['prompt', 'val', 'train']
    results = []
    for prompt, acc in templates:
        results.append([prompt, acc['val'], acc['train']])
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(filename, index=False)


def write_templates_to_csv_llava(templates, filename):
    def format_percent(x):
        if isinstance(x, (float)):
            return '{:.6%}'.format(x)
        else:
            return x
    columns = ['prompt', 'train']
    results = []
    for prompt, acc in templates:
        results.append([prompt, acc['train']])
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(filename, index=False)



dataset_info = {
    'EuroSAT': 'EuroSAT is a dataset based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with in total of 27,000 labeled and geo-referenced images.',
    'OxfordFlowers':  'Oxford Flowers consists of 102 flower categories and is used for the fine-grained visual classification tasks. The flowers are chosen to be flowers that commonly occur in the United Kingdom.',
    'ImageNet': 'ImageNet is a large-scale fine-grained image classification dataset containing over 14 million labeled images across more than 20,000 categories. It has been widely used for training and evaluating computer vision models, particularly in the context of image classification tasks.',
    'UCF101': 'UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. The action categories can be divided into five types: 1)Human-Object Interaction 2) Body-Motion Only 3) Human-Human Interaction 4) Playing Musical Instruments 5) Sports.',
    'ImageNetR': 'ImageNet-R(endition) contains art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos, toys, and video game renditions of ImageNet classes.',
    'ImageNetSketch': 'ImageNet-Sketch data set consists of sketches of the original categories in the ImageNet dataset. The dataset is used for the task of object recognition of the sketched images.',
    'DescribableTextures': 'The Describable Textures Dataset (DTD) is an evolving collection of textural images in the wild, annotated with a series of human-centric attributes, inspired by the perceptual properties of textures. This data is made available to the computer vision community for research purposes.',
    'Food101': 'The Food101 dataset is a fine-grained object recognition dataset containing images of 101 food categories.',
    'FGVCAircraft': 'Fine-grained visual Classification of Aircraft (FGVC-Aircraft) is a benchmark dataset for the fine-grained visual categorization of aircraft.',
    'Caltech101': 'Caltech 101 contains a total of 9,146 images, split between 101 distinct object categories (faces, watches, ants, pianos, etc.) and a background category.',
    'OxfordPets': 'The Oxford-IIIT Pet Dataset is a 37-category pet dataset with roughly 200 images for each class. This dataset is used for the task of image classification.',
    'StanfordCars': 'The Stanford Cars is a fine-grained object classification dataset consisting of finegrained categories of cars.',
    'RESISC45': 'RESISC45 dataset is used for the task of image classification and contains 45 different categories of images that are obtained from satellite imagery, and aerial photography.',
    'ImageNetA': 'ImageNetA is a dataset consisting of natural adversarial examples -- real-world, unmodified, and naturally occurring examples that cause classifier accuracy to significantly degrade.',
    'SUN397': 'The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of 397 categories (e.g., abbey, airplane cabin, athletic field outdoor, atrium public, basilica, canyon, creek, etc).',
    'chartqa': 'chartqa is A Benchmark for Question Answering about Charts with Visual and Logical Reasoning',
    'gqa': 'The GQA dataset is a large-scale visual question answering dataset with real images from the Visual Genome dataset and balanced question-answer pairs. Each training and validation image is also associated with scene graph annotations describing the classes and attributes of those objects in the scene, and their pairwise relations.'
    }


lov_guidance = {
    'EuroSAT': 'Label the image as [one of the 10 classes] based on the prominent features and satellite features present, providing a concise description of the dominant land cover or vegetation type, and highlighting any notable patterns or structures in the image.',
    'OxfordFlowers':  'Identify the type of flower in this image and provide its common name e.g. \'This is a species of [Common Name]\.',
    'ImageNet': 'Can you describe the main subject or object in this image, highlighting its most distinctive visual features, typical attributes, and common name, and explain how it relates to its broader category by tracing its evolution through time, exploring its cultural and historical significance, and highlighting its relationships with other objects within that category, while also emphasizing the subtle nuances and peculiarities that set',
    'ImageNetV2': 'Can you describe the main subject or object in this image, highlighting its most distinctive visual features, typical attributes, and common name, and explain how it relates to its broader category by tracing its evolution through time, exploring its cultural and historical significance, and highlighting its relationships with other objects within that category, while also emphasizing the subtle nuances and peculiarities that set',
    'UCF101': 'Describe the human activity in this image, emphasizing the specific actions, objects, and actors involved, and identify the UCF101 category that best captures this action by highlighting the type of interaction (human-object, body-motion, human-human, or sports) and providing a detailed category name that accurately matches the action depicted, such as \'Human-Object Interaction',
    'ImageNetR': 'Can you describe the visual category depicted in this image by highlighting its creative context, notable features, and artistic medium, and specify the name of the corresponding ImageNet-R class while examining how the artwork reinterprets and recontextualizes the original ImageNet class\'s conventions, incorporating artistic liberties and creative flair',
    'ImageNetSketch': 'Envision the sketched representation of the object, highlighting its distinctive visual patterns, functional relationships with other ImageNet categories, and typical environments, while emphasizing its versatility and common associations, and crafting a nuanced description that accurately integrates its adaptability, potential applications, and versatility, ensuring a precise mention of the class name and corresponding ImageNet category.',
    'DescribableTextures': 'What specific texture category is present in this image, defined by its unique visual cues, spatial frequency, and luminance, as perceived by human observers, and characterized by its distinctive pattern of alternating attributes that vary in terms of roughness, softness, and bumpy or smooth features, while also considering the subtle interactions between these cues, the surrounding context,',
    'Food101': 'Vividly describe the image\'s composition, highlighting the main ingredients, cooking techniques, and presentation styles that make it unique, while specifying the exact category of food and briefly explaining the cultural significance of the dish, focusing on the sensory details that evoke a sense of warmth, comfort, and regional or international influences that shape the culinary tradition.',
    'FGVCAircraft': 'Can you identify the specific aircraft model or subcategory shown in this image, and mention a key distinguishing characteristic that is both visually apparent to a non-expert observer and closely related to the aircraft\'s design evolution or historical context?',
    'Caltech101': 'Classify this image as one of the 101 object categories in the Caltech 101 dataset, by pinpointing the object\'s most salient visual elements and its nuanced interactions with the surrounding environment, while providing a concise and accurate label for its corresponding category name that effectively captures the object\'s proportions, orientation, and subtle context-dependent appearances.',
    'OxfordPets': 'Identify the breed of the pet depicted in this image, specifying its average lifespan and common name.',
    'StanfordCars': 'Classify the image as a specific car model, emphasizing its striking design features, precise manufacturer, exact model year, and notable details, while highlighting the subtle variations in its color palette, trim levels, and overall styling to accurately categorize it among the fine-grained categories of cars.',
    'RESISC45': 'Can you describe the geographical feature or man-made structure depicted in the image, highlighting its unique characteristics, features, and patterns that make it distinct from other categories, and then consider the surrounding environment, terrain, and any notable visual anomalies or textures that provide contextual clues to help identify the category from RESISC45?',
    'ImageNetA': 'Interpret the image as a subtle anomaly within a broader category, where the depicted concept or object\'s distinctive features and deviations from typical expectations subtly alter our understanding of the category\'s identity and necessitate a nuanced classification.',
    'SUN397': 'Envision the scene in this image, where the masterful blend of visual and contextual nuances yields a distinct narrative, thoughtfully guiding you to intuit the specific category from the 397 SUN categories, with precision and attention to the intricate relationships that harmonize to define the scene\'s membership within its designated category, while subtly illuminating the most salient and characteristic.',
    }
lov = {
    'EuroSAT': 'Label the image as [one of the 10 classes] based on the prominent features and satellite features present, providing a concise description of the dominant land cover or vegetation type, and highlighting any notable patterns or structures in the image.',
    'OxfordFlowers':  'Identify the specific type of flower depicted in this image, providing its botanical name and a detailed description of its unique characteristics, including its color palette, shape, texture, and any distinctive markings or patterns, while highlighting its botanical classification and the ways in which it has evolved to occupy a specific ecological niche in the diverse habitats and temperate maritime climate of',
    'ImageNet': 'Spot the distinctive visual cues, textures, or patterns in this image, linking them to the exact class name, while also considering the contextual elements that help disambiguate it from similar classes.',
    'ImageNetV2': 'Spot the distinctive visual cues, textures, or patterns in this image, linking them to the exact class name, while also considering the contextual elements that help disambiguate it from similar classes.',
    'UCF101': 'Elaborate on the specific attributes and characteristics of the human or object in the image that uniquely define the UCF101 action category, highlighting notable patterns, shapes, or movements that distinguish it from others, and further describe the context and scene where the action takes place.',
    'ImageNetR': 'Can you describe the visual category depicted in this image, weaving together artistic expression, cultural context, and semantic meaning to specify the ImageNet-R class that masterfully harmonizes creative and literal aspects of the depiction, while acknowledging the nuanced interplay between artistic interpretation, cultural influences, and original meaning in the representation?',
    'ImageNetSketch': 'Envision the original ImageNet object\'s most distinctive attributes and describe how the sketched representation masterfully captures these nuances, ensuring a precise correspondence to the class name.',
    'DescribableTextures': 'Identify the texture category and describe its characteristic visual pattern, emphasizing the striking visual cues that make it instantly recognizable within its category, while highlighting the most prominent feature that sets it apart from others.',
    'Food101': 'What type of food is depicted in this image, and how do its visual features, cultural associations, and culinary traditions coalesce to create a rich culinary identity that encompasses its unique flavors, cultural significance, and appeal?',
    'FGVCAircraft': 'Pinpoint the aircraft model, emphasizing its distinctive configuration of wings, fuselage, and control surfaces, while highlighting the nuanced variations that differentiate it from other models within the broader category of aircraft, and accurately distinguishing it from similar models.',
    'Caltech101': 'This object is a paradigmatic instance of [Caltech category name], exemplifying the core characteristics and features that define the concept and accurately capturing the essence of its category.',
    'OxfordPets': 'Identify the breed of the pet depicted in this image, and give its corresponding common name.',
    'StanfordCars': 'Describe the specific make and model of the car in the image, highlighting its unique design elements, notable features, and overall aesthetic appeal, while also analyzing its market positioning, technological advancements, and historical significance within the automotive industry, ultimately revealing its distinctiveness within its class.',
    'RESISC45': 'Can you describe the satellite or aerial photograph by focusing on the distinct spatial relationships and arrangements of geographical features or man-made structures that define its category, and then categorize it into one of the 45 categories in the RESISC45 dataset by emphasizing the unique characteristics that set it apart from other categories while considering the contextual information provided?',
    'ImageNetA': 'Describe the object or concept depicted in this image by highlighting the most significant visual cues that deviate from typical representations, and identify the category name while emphasizing the subtle differences between this instance and expected examples within the same class.',
    'SUN397': 'Classify the scene in this image by teasing out its intricate essence through a nuanced analysis of its visual topography, comprising the harmonious interplay of its most prominent elements, spatial arrangements, and subtle contextual cues, thereby pinpointing the precise SUN category that accurately captures its unique character and situates it within the 397 options.',
    }


lov_foci = { 
    'FGVCAircraft' : 'Can you describe the aircraft model and manufacturer depicted in this image, highlighting its most distinctive features and unique design elements that distinguish it from other similar models?',
    'OxfordPets' : 'What OxfordPets breed is this image most likely to belong to, considering the visual characteristics and features described in the Oxford-IIIT Pet Dataset?',
    'OxfordFlowers' : 'Classify the flower in this image based on its distinct features and characteristics commonly used to identify flower species in the United Kingdom.',
    'Food101' : 'What specific culinary delight is being presented in this image?',
    }
lov_guidance_foci = {
    'FGVCAircraft' : 'What aircraft model is depicted in this image, showcasing its unique design features, era of service, and remarkable feats in aviation, to accurately identify the specific aircraft model?',
    'OxfordPets' : 'What OxfordPets breed is highlighted in this image, and how does its distinctive appearance and characteristics contrast with those of other breeds?',
    'OxfordFlowers' : 'Can you please classify the flower species in this image, noting its genus and key features, and highlighting its unique characteristics that distinguish it from its closest relatives within the same genus while also specifying its exact category within the 102 types of flowers?',
    'Food101' : 'What food is being served in this image, considering its textures, colors, and culinary and cultural context, as well as its typical preparation and serving methods?',
    }


meta_prompt = {
    'EuroSAT':'Label the image as [one of the 10 classes] based on the dominant land cover or vegetation type shown.',
    'OxfordFlowers':'',
    'ImageNet':'Can you identify the main object in this image and provide its common name?',
    'ImageNetV2':'Can you identify the main object in this image and provide its common name?',
    'UCF101': 'In this image, an individual is performing ___________________________, which is a typical example of the ___________________________ category.',
    'ImageNetR':'Can you describe the visual category depicted in this image, mentioning the style, medium, and any notable features, and specify the name of this ImageNet-R class?',
    'ImageNetSketch':'Identify the sketched object and describe its purpose or function, ensuring the category name is accurately mentioned.',
    'DescribableTextures':'Identify the texture category and describe it in a few words, using attributes like rough, smooth, soft, hard, bumpy, etc.',
    'Food101':'Name the specific type of food shown in this image, such as \'pizza\' or \'sushi\', and provide a brief description of its characteristics.',
    'FGVCAircraft':'',
    'Caltech101':'Identify the object shown in the image and provide its corresponding category name, like \'watch\' for a timepiece or \'ant\' for a small insect, etc.',
    'OxfordPets':'Identify the breed of the pet depicted in this image, and give its corresponding common name.',
    'StanfordCars':'Describe the type of car in the image and provide its exact model name, if possible.',
    'RESISC45':'Identify the geographical feature or man-made structure present in the image, and briefly describe it, then state the corresponding category from RESISC45.',
    'ImageNetA':'Explain the concept or object depicted in this image and how it differs from a typical instance of the same category, including its name.',
    'SUN397':''
    }

text_cls_epochs = {
    'DescribableTextures': 3000,
    'EuroSAT': 400,
    'FGVCAircraft': 500,
    'Food101': 400,
    'OxfordFlowers': 600,
    'SUN397': 2000,
    'UCF101': 1050,
    'ImageNetR': 5000,
}

def setup_txt_epochs(args, dataset):
    args.txt_epochs = text_cls_epochs[dataset]


def get_env_id():
    return getpass.getuser()


def test_wo_prompting(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, inputs in enumerate(tqdm(teloader)):
        labels = inputs['label']
        inputs = inputs['img']
        if isinstance(inputs, list):
            inputs = inputs[0]

        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.eval_clip_wo_prompt(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DINOLoss(nn.Module):
    def __init__(self, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            loss = torch.sum(-q * F.log_softmax(student_out, dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


def setup_log_folder_(args):
    Path(args.logfolder).mkdir(exist_ok=True, parents=True)
    args.logfile = args.logfolder + f'/{time.strftime("%Y%m%d_%H%M%S")}.txt'

def setup_log_folder(args, base_folder):
    args.logfile = f'{base_folder}/logs.txt'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def zero_shot(model, loader):
    print('-------------- ZERO SHOT INFERENCE --------------')
    total = 0.
    correct_base = 0.
    model.eval()





    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            target = inputs['label']
            images = inputs['img']
            if isinstance(images, list):
                images = images[0]

            images = images.cuda()
            target = target.cuda()
            out = model(images)
            logits_base = out
            pred_base = torch.argmax(logits_base, dim=1)
            for j in range(len(target)):
                total += 1.
                if pred_base[j] == target[j]:
                    correct_base += 1.
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy standard: {top1:.2f}")


def zero_shot_stupid(model, loader):
    print('-------------- ZERO SHOT INFERENCE --------------')
    total = 0.
    correct_base = 0.
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            target = inputs['label']
            images = inputs['img']
            if isinstance(images, list):
                images = images[0]

            images = images.cuda()
            target = target.cuda()
            out = model(images)
            logits_base = out
            pred_base = torch.argmax(logits_base, dim=1)
            for j in range(len(target)):
                total += 1.
                if pred_base[j] == target[j]:
                    correct_base += 1.
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy standard: {top1:.2f}")

def new_zero_shot(model, loader):
    print('-------------- ZERO SHOT INFERENCE --------------')
    total = 0.
    correct_base = 0.
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            target = inputs['label']
            images = inputs['img']
            if isinstance(images, list):
                images = images[0]

            images = images.cuda()
            target = target.cuda()
            out = model.new_zero_shot(images)
            logits_base = out
            pred_base = torch.argmax(logits_base, dim=1)

            for j in range(len(target)):
                total += 1.
                if pred_base[j] == target[j]:
                    correct_base += 1.
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy standard: {top1:.2f}")


def zero_shot_c10_100_imagenet(model, loader):
    # print('herrr')
    # quit()
    print('-------------- ZERO SHOT INFERENCE --------------')
    total = 0.
    correct_base = 0.
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            target = inputs[1]
            images = inputs[0]
            if isinstance(images, list):
                images = images[0]
            images = images.cuda()
            target = target.cuda()
            # predict
            out = model(images)
            logits_base = out
            pred_base = torch.argmax(logits_base, dim=1)
            for j in range(len(target)):
                total += 1.
                if pred_base[j] == target[j]:
                    correct_base += 1.
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy standard: {top1:.2f}")


def test(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, (inputs, labels) in enumerate(tqdm(teloader)):
        if isinstance(inputs, list):
            inputs = inputs[0]

        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.eval_clip(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100


def test_prompting(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, inputs in enumerate(tqdm(teloader)):
        labels = inputs['label']
        inputs = inputs['img']
        if isinstance(inputs, list):
            inputs = inputs[0]

        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.eval_clip(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100


def test_prompting_c10_100_imagenet(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, (inputs, labels) in enumerate(tqdm(teloader)):
        # labels = labels.cuda()
        # inputs = inputs.cuda()
        if isinstance(inputs, list):
            inputs = inputs[0]
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.eval_clip(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100


def test_prompting_text(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, (inputs, labels) in enumerate(tqdm(teloader)):
        if isinstance(inputs, list):
            inputs = inputs[0]

        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.forward_prompt_inf(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100


def update_ema_variables(model, alpha_teacher):
    teacher_prompt_param = []
    student_prompt_param = []

    for key, value in model.named_parameters():
        if key == 'prompt_embeddings':
            student_prompt_param.append(value)

        elif key == 'prompt_embeddings_teacher':
            teacher_prompt_param.append(value)

    for ema_param, param in zip(teacher_prompt_param, student_prompt_param):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[
                                                                                         :]  # alpha * teacher_weights + (1 - alpha) * student_weights

    for k, v in model.named_parameters():
        if k == 'prompt_embeddings_teacher':
            v = ema_param

    # return ema_model


def update_ema_variables_sanity(ema_model, model, alpha_teacher):
    for kv_ema, kv_student in zip(ema_model.named_parameters(), model.named_parameters()):
        if 'ln' in kv_ema[0] and 'ln' in kv_student[0]:
            kv_ema[1].data[:] = alpha_teacher * kv_ema[1][:].data[:] + (1 - alpha_teacher) * kv_student[1][:].data[:]
    return ema_model


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            last_epoch=-1,
            verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            cons_lr,
            last_epoch=-1,
            verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


def setup_prompt_training_utils(args, model):
    model = model.cuda()
    model = model.float()
    params = list()
    print('Mile Stones: ', args.mile_stones)

    # mile_stones = args.mile_stones = None
    # mile_stones = np.arange(5, args.epochs, 5)
    for key, value in model.named_parameters():
        if key == 'prompt_embeddings':
            value.requires_grad = True
        elif 'adapter' in key:
            value.requires_grad = True
        elif 'projector' in key and not args.entropy:
            value.requires_grad = True
        elif 'ln' in key:
            value.requires_grad = True
        else:
            value.requires_grad = False

    for key, value in model.named_parameters():
        if 'visual' in key:
            if 'ln' in key or 'bn' in key:
                value.requires_grad = True
            else:
                value.requires_grad = False

    if args.deep_prompt:
        for key, value in model.named_parameters():
            if key == 'deep_prompt_embeddings':
                value.requires_grad = True

    print('------------------ Learnable Parameters ------------------')
    for key, value in model.named_parameters():
        if value.requires_grad:
            print("\t{}, {}, {}".format(key, value.numel(), value.shape))
            params.append((key, value))
    print('----------------------------------------------------------')

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.mile_stones, 0.60)
    criteria = LabelSmoothingCrossEntropy()
    return optimizer, scheduler, criteria


def setup_prompt_training_utils_sanity(args, model, model_pl, log):
    model = model.cuda()
    model = model.float()
    params = list()
    mile_stones = args.mile_stones

    for k, v in model_pl.named_parameters():  # for sanity FREEZE the pseudo-label model
        v.requires_grad = False

    for key, value in model.named_parameters():
        if key == 'prompt_embeddings':
            value.requires_grad = True
        elif 'adapter' in key and not args.entropy:
            value.requires_grad = True
        elif 'ln' in key:
            value.requires_grad = True
        else:
            value.requires_grad = False
    log.info('------------------ Learnable Parameters ------------------')
    for key, value in model.named_parameters():
        if value.requires_grad:
            log.info("\t{}, {}, {}".format(key, value.numel(), value.shape))
            params.append((key, value))
    log.info('----------------------------------------------------------')

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999))

    if args.scheduler == 'coslr':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=args.epochs,
                                      lr_min=1e-6,
                                      warmup_lr_init=1e-4,
                                      warmup_t=5,
                                      cycle_limit=1,
                                      t_in_epochs=True)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, 0.1)

    criteria = LabelSmoothingCrossEntropy()
    return optimizer, scheduler, criteria


def setup_prompt_training_utils_text(args, model, log):
    model = model.cuda()
    model = model.float()
    params = list()
    mile_stones = args.mile_stones
    name_to_update = "prompt_learner"

    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad_(True)
            elif 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad_(False)

    log.info('------------------ Learnable Parameters ------------------')
    for key, value in model.named_parameters():
        if value.requires_grad:
            log.info("\t{}, {}, {}".format(key, value.numel(), value.shape))
            params.append((key, value))
    log.info('----------------------------------------------------------')

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999))

    if args.scheduler == 'coslr':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=args.epochs,
                                      lr_min=1e-6,
                                      warmup_lr_init=1e-4,
                                      warmup_t=5,
                                      cycle_limit=1,
                                      t_in_epochs=True)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, 0.1)

    criteria = LabelSmoothingCrossEntropy()
    return optimizer, scheduler, criteria


def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger



def save_args(path, args):

    file_path = Path(path + '/args.txt') 

    args_dict = vars(args)

    # Write the arguments to the file
    with open(file_path, 'w') as f:
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")

    print(f"Arguments saved to {file_path}")
