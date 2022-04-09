import warnings
warnings.filterwarnings("ignore")

import os
import random
import argparse
import numpy as np

from typing import Union

import umap
import matplotlib.pyplot as plt
from matplotlib import cm

import librosa
import librosa.display

import io
import PIL.Image
from torchvision.transforms import ToTensor

from glob import glob

from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.tensorboard import TensorBoardCallback

#-----------------------------
PAR_N_FRAMES = 160  
MEL_N_CHANNELS = 40 

EM_SIZE = 256

FIGSIZE = (21, 12)

CMAP = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255

MAX_SPKR = 10

SUPER = {
    "size": 18,
    "weight": 600,
}
#---------------------------------------------------------
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")

parser = argparse.ArgumentParser()

parser.add_argument("-frun", "--fast_run", "--fast", type=bool, default=False)
parser.add_argument("-e", "--epochs", "--num_epochs", type=int, default=20)

parser.add_argument("-tlim", "--per_train", "--train_limit",  default="30")
parser.add_argument("-vlim", "--per_val", "--val_limit", default="20")

parser.add_argument("-v", "--version_name", "--vname", default="default")

parser.add_argument("-nwrkr", "--num_workers", default=os.cpu_count())

parser.add_argument("-tune", "--istuning", action='store_true')

parser.add_argument("-lrate", "--learning_rate", default=1e-4)
parser.add_argument("-nlayers", "--num_layers", default=3)
parser.add_argument("-hlsize", "--hidden_layer_size", type=int, default=256)

parser.add_argument("-spkr", "--speaker_per_batch", "--spkr_per_batch", type=int,default=8) # should be 64
parser.add_argument("-nutter", "--utterances_per_speaker", "--utter_per_spkr", type=int, default=5) # should be 10

parser.add_argument("-dir", "--data_dir", "--directory",  type=dir_path, default="../data/audio")

args = parser.parse_args()

args.per_train = int(args.per_train) if args.per_train.isdigit() else float(args.per_train)
args.per_val = int(args.per_val) if args.per_val.isdigit() else float(args.per_val)

istraining = not args.istuning 

nchoice = args.epochs if args.epochs < 10 else (np.random.randint(args.epochs//5, args.epochs-5) if args.epochs <20 else 16)

nplot = (2, 1) if nchoice==2 else ((nchoice/2, 2) if nchoice<7 else ((3, nchoice/3) if nchoice<10 else (4, nchoice/4)))
nplot = (int(np.ceil(nplot[0])), int(np.ceil(nplot[1])))

epoch_idxs = list(range(args.epochs)) if args.epochs<10 else np.random.choice(list(range(1, args.epochs)), nchoice, False)
#---------------------------------------------------------------
class RandomCycler:
    def __init__(self, source):
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count):
        shuffle = lambda l: random.sample(l, len(l))
        
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
        return out

    def __next__(self):
        return self.sample(1)[0]

class Utterance:
    def __init__(self, frames_fpath, ):
        self.frames_fpath = frames_fpath
        
    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        frames = self.get_frames()
        if frames.shape[0] == n_frames: start = 0
        else: start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)

class Speaker:
    def __init__(self, root):
        self.root = root
        self.utterances = None
        self.utterance_cycler = None
        
    def load_utterances(self): 
        self.utterances = [Utterance(upath) for upath in glob(self.root + "/*")]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count, n_frames):
        if self.utterances is None: self.load_utterances()

        utterances = self.utterance_cycler.sample(count)

        return [(u,) + u.random_partial(n_frames) for u in utterances]

class SpeakerBatch:
    def __init__(self, speakers, utterances_per_speaker, n_frames):
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])

class SpeakerVerificationDataset(Dataset):
    def __init__(self, dataset_root):
        self.root = dataset_root
        speaker_dirs = [f for f in glob(self.root + f"/*")]
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)

class SpeakerVerificationDataLoader(DataLoader):
    def __init__(
            self, 
            dataset, 
            speakers_per_batch, 
            utterances_per_speaker, 
            sampler=None, 
            batch_sampler=None, 
            num_workers=0, 
            pin_memory=False, 
            timeout=0, 
            worker_init_fn=None
        ):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, n_frames=PAR_N_FRAMES)

#---------------------------------------------------------------------------------------------
class SpeakerVerificationDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir=None, test_dir=None):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
    
    @staticmethod
    def create_dataloader(path):
        return SpeakerVerificationDataLoader(
            SpeakerVerificationDataset(path),
            speakers_per_batch=args.speaker_per_batch,
            utterances_per_speaker=args.utterances_per_speaker,
            num_workers=args.num_workers,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_dir)
    
    def val_dataloader(self):
        return self.create_dataloader(self.val_dir)

    def test_dataloader(self):
        return self.create_dataloader(self.test_dir)

# ----------------------------------------------------------------------------------
class SpeakerEncoder(pl.LightningModule):
    def __init__(self, 
            hidden_layer_size=args.hidden_layer_size, 
            n_layers=args.num_layers, 
            learning_rate=args.learning_rate
        ):
        super().__init__()   
        self.learning_rate = learning_rate 
        self.metrics = {
            "loss": [],
            "eer": [],
            "val_loss": [],
            "val_eer": []
        }

        self.data = {
            "metric/loss": [],
            "metric/eer": [],
            "metric/val_loss": [],
            "metric/val_eer": [],
            "input/embeds": [],
            
            "input/proj": {
                "vector": [],
                "meta": "Speaker\n"
            }

        }

        self.meta = "labelc1\tlabelc2\tlabelc3\t\n"

        # Network defition
        self.lstm = nn.LSTM(
            input_size=MEL_N_CHANNELS,
            hidden_size=hidden_layer_size, 
            num_layers=n_layers, 
            batch_first=True
        )
        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=EM_SIZE)
        self.relu = torch.nn.ReLU()
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, utterances, hidden_init=None):
        _, (hidden, _) = self.lstm(utterances, hidden_init)
        
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds
    
    def similarity_matrix(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int32)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
        
    def loss(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long()
        loss = self.loss_fn(sim_matrix, target)
        
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int32)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def common_step_batch(self, batch, batch_idx, stage=None):
        inputs = torch.from_numpy(batch.data)        

        embeds = self(inputs)
        embeds_loss = embeds.view((args.speaker_per_batch, args.utterances_per_speaker, -1))

        loss, eer = self.loss(embeds_loss)

        if (istraining) and stage=="train":

            if self.current_epoch in epoch_idxs: 
                idx = np.random.randint(1, len(inputs))

                height = int(np.sqrt(len(embeds[idx])))
                shape = (height, -1)
                self.data["input/embeds"].append(
                    embeds[idx].reshape(shape).detach().numpy()
                )

                cur_embeds = embeds.detach()
                cur_embeds = cur_embeds[:MAX_SPKR * args.utterances_per_speaker]

                self.data["input/proj"]["vector"].append(cur_embeds)

                n_speakers = len(cur_embeds ) // args.utterances_per_speaker
                ground_truth = np.repeat(np.arange(n_speakers), args.utterances_per_speaker)

                self.data["input/proj"]["meta"] += "/n".join([f"speaker{i}" for i in ground_truth])
                           
            if self.current_epoch==0:
                idxs = np.random.choice(range(1, len(inputs)), 16, False)

                self.logger.experiment.add_image(
                    "plots/inputs/mel_specs",
                    SpeakerEncoder.input_to_image(inputs[idxs],
                        "random Melody-Spectrogram pool",
                        "Time", "Mels"
                    ), 0, dataformats='NCHW'
                )

                self.logger.experiment.add_graph(SpeakerEncoder(), inputs)

        return loss, eer

    def training_step(self, batch, batch_idx):
        loss, eer = self.common_step_batch(batch, batch_idx, "train")
        return {
            "loss": loss,
            "eer": eer
        }

    def validation_step(self, batch, batch_idx):
        loss, eer = self.common_step_batch(batch, batch_idx, "val")
        return {
            "val_loss": loss,
            "val_eer": eer
        }

    def training_epoch_end(self, outputs):
        loss = [x["loss"].detach().item() for x in outputs]
        eer = [x["eer"] for x in outputs]

        if self.current_epoch in epoch_idxs: 
            self.data["metric/loss"].append(loss)
            self.data["metric/eer"].append(eer)

        loss = np.mean(loss)
        eer = np.mean(eer)

        self.metrics["loss"].append(loss)
        self.logger.experiment.add_scalar("AvgLoss/train", loss, self.current_epoch)

        self.metrics["eer"].append(eer)
        self.logger.experiment.add_scalar("AvgEER/train", eer, self.current_epoch)    

    def validation_epoch_end(self, outputs):
        loss = [x["val_loss"].detach().item() for x in outputs]
        eer = [x["val_eer"] for x in outputs]
        
        if self.current_epoch in epoch_idxs: 
            self.data["metric/val_loss"].append(loss)
            self.data["metric/val_eer"].append(loss)

        loss = np.mean(loss)
        eer = np.mean(eer)

        self.metrics["val_loss"].append(loss)
        self.logger.experiment.add_scalar("AvgLoss/val", loss, self.current_epoch)

        self.metrics["val_eer"].append(eer)
        self.logger.experiment.add_scalar("AvgEER/val", eer, self.current_epoch)

    def on_sanity_check_end(self):
        self.metrics = {}

    @staticmethod
    def metric_to_image(data, title, xlabel, ylabel):
        fig, ax = plt.subplots(nplot[0], nplot[1], figsize=FIGSIZE)
        fig.add_subplot(111, frameon=False)
        fig.suptitle(title, size=SUPER["size"], weight=SUPER["weight"])
        count = 0
        for m, n in [(i, j) for i in range(nplot[0]) for j in  range(nplot[1])]:

            if (nplot[0] != 1) and (nplot[1] != 1): cur_ax = ax[m, n]
            else: cur_ax = ax[count]

            try:
                y = data[count]
                cur_ax.plot(y)

                if m+1 != nplot[0]: cur_ax.set_xticks([])
            except: 
                cur_ax.set_xticks([])
                
            cur_ax.set_yticks([])
                
            count += 1

        SpeakerEncoder.add_labels(xlabel, ylabel)
        return SpeakerEncoder.figure_to_image()
    
    @staticmethod
    def input_to_image(data, title, xlabel, ylabel, rows=4, cols=4):
        fig, ax = plt.subplots(rows, cols, figsize=FIGSIZE)
        fig.add_subplot(111, frameon=False)
        fig.suptitle(title, size=SUPER["size"], weight=SUPER["weight"])
        count = 0
        for m,n in [(i, j) for i in range(rows) for j in  range(cols)]:
            img = librosa.display.specshow(
                librosa.power_to_db(data[count].T, ref=np.max), 
                fmax=8000, ax=ax[m, n], y_axis="mel", x_axis="time"
            )

            if m+1 != rows: ax[m, n].set_xticks([])
            if n!=0: ax[m, n].set_yticks([])

            ax[m, n].set_xlabel("")
            ax[m, n].set_ylabel("")

            count += 1

        SpeakerEncoder.add_labels(xlabel, ylabel)
        return SpeakerEncoder.figure_to_image()

    @staticmethod
    def embeds_to_image(data, title):
        plt.clf()

        fig, ax = plt.subplots(nplot[0], nplot[1], figsize=(nplot[0]*4, nplot[1]*4))
        fig.add_subplot(111, frameon=False)
        fig.suptitle(title, size=SUPER["size"], weight=SUPER["weight"])
        count = 0
        for m, n in [(i, j) for i in range(nplot[0]) for j in  range(nplot[1])]:

            if (nplot[0] != 1) and (nplot[1] != 1): cur_ax = ax[m, n]
            else: cur_ax = ax[count]

            try:
                cmap = cm.get_cmap()
                cur_ax.imshow(data[count], cmap=cmap)
            except: pass

            cur_ax.set_xticks([])    
            cur_ax.set_yticks([])

            count += 1

        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        return SpeakerEncoder.figure_to_image()

    @staticmethod
    def add_labels(xlabel=None, ylabel=None, label_size=14, padx=5, pady=2):        
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)

        plt.xlabel(xlabel, labelpad=padx, size=label_size)
        plt.ylabel(ylabel, labelpad=pady, size=label_size)

        plt.tight_layout()

    @staticmethod
    def figure_to_image():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        image = PIL.Image.open(buf)
        image = np.array(image)[...,:3]
        image = ToTensor()(image).unsqueeze(0)
        return image

class SpeakerEncoderCallbacks(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        # Gradient scale
        pl_module.similarity_weight.grad *= 0.01
        pl_module.similarity_bias.grad *= 0.01          
        # Gradient clipping
        clip_grad_norm_(pl_module.parameters(), 3, norm_type=2)

    def on_fit_end(self, trainer, pl_module):
        trainer.callback_metrics["loss"] = np.mean(pl_module.metrics["loss"])
        trainer.callback_metrics["val_loss"] = np.mean(pl_module.metrics["val_loss"])

        trainer.callback_metrics["eer"] = np.mean(pl_module.metrics["eer"])
        trainer.callback_metrics["val_eer"] = np.mean(pl_module.metrics["val_eer"]) 

        pl_module.logger.experiment.add_image(
            "plots/metrics/loss", 
            SpeakerEncoder.metric_to_image(
                pl_module.data["metric/loss"], 
                "Training Loss by Step per Epoch",
                "step (batch-number)",
                "loss (GE2E loss)"
            ), 0, dataformats='NCHW'
        )

        pl_module.logger.experiment.add_image(
            "plots/metrics/eer", 
            SpeakerEncoder.metric_to_image(
                pl_module.data["metric/eer"], 
                "Training EER by Step per Epoch",
                "step (batch-number)",
                "EER (Equal Error Rate)"
            ), 0, dataformats='NCHW'
        )

        pl_module.logger.experiment.add_image(
            "plots/metrics/val_loss", 
            SpeakerEncoder.metric_to_image(
                pl_module.data["metric/val_loss"], 
                "Validation Loss by Step per Epoch",
                "step (batch-number)",
                "val loss (GE2E loss)"
            ), 0, dataformats='NCHW'
        )

        pl_module.logger.experiment.add_image(
            "plots/metrics/val_eer", 
            SpeakerEncoder.metric_to_image(
                pl_module.data["metric/val_eer"], 
                "Validation EER by Step per Epoch",
                "step (batch-number)",
                "EER (Equal Error Rate)"
            ), 0, dataformats='NCHW'
        )

        if istraining:
            pl_module.logger.experiment.add_image(
                "plots/input/embeds", 
                SpeakerEncoder.embeds_to_image(
                    pl_module.data["input/embeds"], 
                    "random Embeddings pool",
                ), 0, dataformats='NCHW'
            )

            pl_module.logger.experiment.add_embedding(
                torch.stack(pl_module.data["input/proj"]["vector"]), 
                metadata = pl_module.data["input/proj"]["meta"], 
                tag = f"default"
            )


if __name__=="__main__":
    warnings.filterwarnings("ignore")

    ## -------------- TRAINING LOOP ------------------------
    datamodule = SpeakerVerificationDataModule(f"{args.data_dir}/train", f"{args.data_dir}/val")

    model = SpeakerEncoder()   
    model_cb = SpeakerEncoderCallbacks()

    tb_logger = pl.loggers.TensorBoardLogger("../lightning_logs", args.version_name)

    trainer = pl.Trainer(callbacks=[model_cb],
                         logger=tb_logger,
                         fast_dev_run=args.fast_run, 
                         max_epochs=args.epochs,
                         gpus=None,
                         tpu_cores=None,
                         limit_train_batches=args.per_train,
                         limit_val_batches=args.per_val,
                        )

    trainer.fit(model, datamodule=datamodule)

    print(trainer.callback_metrics)

