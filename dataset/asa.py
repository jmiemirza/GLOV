import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
from ImageBind.imagebind.data import *
import csv
from dassl.utils import listdir_nohidden, mkdir_if_missing
import os.path as osp
import glob
IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


def read_csv_to_list(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

@DATASET_REGISTRY.register()
class esc(DatasetBase):
    def __init__(self, cfg):
        # super().__init__(cfg)

        # with open('/data1/llm_as_optimizer/datasets/esc_cls.txt', 'r') as f:
        #     self.classes = f.read().splitlines()

        data_dir = '/data1/llm_as_optimizer/data/audioset'

        # self.classes = os.walk(data_dir)

        self.classes = sorted(f.name for f in os.scandir(data_dir) if f.is_dir())


        test = self._read_data_test(data_dir)

        super().__init__(train_x=test, test=test)


    def _read_data_test(self, data_dir):
        items = []
        for label, class_name in enumerate(self.classes):
            imnames = glob.glob(data_dir + f'/*.wav')
            for imname in imnames:
                item = Datum(impath=imname, label=label, classname=class_name)
                items.append(item)
        return items



def load_and_transform_audio_data(
    audio_paths,
    device,
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

    for audio_path in tqdm(audio_paths):
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

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)