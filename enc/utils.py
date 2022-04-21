#------ NECESSARY LIBRARIES ------------------------------------
import os
import random
import argparse
import numpy as np

import time
import logging

from glob import glob
from datetime import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

#------ GLOBAL VARS -----------------------------------
PAR_N_FRAMES = 160  
MEL_N_CHANNELS = 40 

EM_SIZE = 256

MAX_SPKR = 10

#------ ARGUMENTS ------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("-nwrkr", "--num_workers", default=os.cpu_count())
parser.add_argument("-frun", "--fast_run", "--fast", type=bool, default=False)
parser.add_argument("-v", "--version_name", default="default")
parser.add_argument("-e", "--epochs", type=int, default=20)

parser.add_argument("-tlim", "--per_train", "--train_limit",  default="30")
parser.add_argument("-vlim", "--per_val", "--val_limit", default="20")

parser.add_argument("-spkr", "--speaker_per_batch", "--spkr_per_batch", type=int,default=8) # should be 64
parser.add_argument("-utter", "--utterances_per_speaker", "--utter_per_spkr", type=int, default=5) # should be 10

parser.add_argument("-lrate", "--learning_rate", default=1e-4)
parser.add_argument("-nlayers", "--num_layers", default=3)
parser.add_argument("-hlsize", "--hidden_layer_size", type=int, default=256)

args = parser.parse_args()

args.per_train = int(args.per_train) if args.per_train.isdigit() else float(args.per_train)
args.per_val = int(args.per_val) if args.per_val.isdigit() else float(args.per_val)

print("Arguments:")
for k, v in args.__dict__.items(): print(f"\t{k}: {v}")
time.sleep(2)

#------ LOGGING --------------------------------------------------------
LOG_FORMAT = '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s'

LOG_FNAME = f"{dt.today().day:02d}-{dt.today().month:02d}-{dt.today().year}.log"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

ROOT = "../data/audio/train"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

file_handler = logging.FileHandler(f"../log/{LOG_FNAME}")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger=logging.getLogger("default")

#------ Utils: FUNCTIONS -------------------------------------------------
lst2str = lambda lst,delimiter=", ": f"{delimiter}".join(list(map(str, lst)))

def log(logger, msg, level="info"):
    getattr(logger, level)(msg)
    time.sleep(0.75)

def save_to_tsv(data, fpath):      
    with open(fpath, "a") as fh:
        if isinstance(data, list): 
            fh.write("\n".join(data))
        else: fh.write(data)
        fh.write("\n") 

#------ Utils: CLASSES ----------------------------------------------------
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

#----------- DATAMODULE ------------------------------------------------------------------
class SpeakerVerificationDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
    
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

#--------- MODEL -------------------------------------------------------------------------
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

            if (self.current_epoch==0) and (batch_idx==1):

                self.logger.experiment.add_graph(SpeakerEncoder(), inputs)

                # Mel-Spectrogram
                idxs = np.random.choice(range(1, len(inputs)), 16, False)

                self.logger.experiment.add_image(
                    "plots/inputs/mel_specs",
                    input_to_image(inputs[idxs],
                        "Random Melody-Spectrogram Pool",
                        "Time", "Mels"
                    ), 0, dataformats='NCHW'
                )


                ## Projections
                if projecting:
                    cur_embeds = embeds[:MAX_SPKR * args.utterances_per_speaker].detach().numpy()

                    save_to_tsv(
                        ["\t".join([str(i) for i in x]) for x in cur_embeds],
                        f"{self.logger.log_dir}/vectors.tsv",
                    )
                    
                    n_speakers = len(cur_embeds ) // args.utterances_per_speaker
                    ground_truth = np.repeat(np.arange(n_speakers), args.utterances_per_speaker)

                    save_to_tsv(
                        [str(i) for i in ground_truth],
                        f"{self.logger.log_dir}/labels.tsv",
                    )
                    
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

        if not self.trainer.sanity_checking:
            self.data["metric/val_loss"].append(loss)
            self.data["metric/val_eer"].append(eer)

        loss = np.mean(loss)
        eer = np.mean(eer)

        self.metrics["val_loss"].append(loss)
        self.logger.experiment.add_scalar("AvgLoss/val", loss, self.current_epoch)

        self.metrics["val_eer"].append(eer)
        self.logger.experiment.add_scalar("AvgEER/val", eer, self.current_epoch)
 
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

        if tuning:
            self.logger.log_hyperparams(self.hparams, metrics={
                "hp_metric": trainer.callback_metrics["val_eer"]
            })

        if projecting:      
            pl_module.logger.experiment.add_embedding(
                np.loadtxt(f"{pl_module.logger.log_dir}/vectors.tsv", delimiter="\t"), 
                np.loadtxt(f"{pl_module.logger.log_dir}/labels.tsv", delimiter="\t"),
                tag = f"{args.version_name}"
            )