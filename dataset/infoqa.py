import math
import random
import os.path as osp

from dassl.utils import listdir_nohidden
import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
# from ..build import DATASET_REGISTRY
# from ..base_dataset import Datum, DatasetBase
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# import cifar_classes as cifar_classes

@DATASET_REGISTRY.register()
class infoqa(DatasetBase):
    def __init__(self, cfg):
        pass