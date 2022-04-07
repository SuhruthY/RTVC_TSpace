import warnings
warnings.filterwarnings("ignore")

import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

SPKR_PER_BATCH = 8 #64
UTTER_PER_SPKR = 5 #10

NUM_WRKR = 10

LRATE = 1e-4
HL_SIZE = 256
EM_SIZE = 256
NUM_LAYERS = 3

# EPOCHS = 20
# PER_TRAIN = 30
# PER_VALID = 20

SUB_TITLED = True
#---------------------------------------------------------
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")

parser = argparse.ArgumentParser()

parser.add_argument("-frun", "--fast_run", "--fast", type=bool, default=False)
parser.add_argument("-e", "--epochs", "--num_epochs", type=int, default=20)
parser.add_argument("-tlim", "--per_train", "--train_limit", type=int, default=30)
parser.add_argument("-vlim", "--per_val", "--val_limit", type=int, default=20)
parser.add_argument("-v", "--version_name", "--vname", default="default")
parser.add_argument("-dir", "--data_dir", "--directory",  type=dir_path, default="../data/audio")

args = parser.parse_args()
# ------------------------------------------------------------------------
class subPlotter:
    def __init__(self, nepoch=args.epochs):
        self.count = 0

        self.nchoice = nepoch if nepoch < 10 else (np.random.randint(nepoch//5, nepoch-5) if nepoch <=20 else 16)

        nplot = ( self.nchoice/2, 2) if  self.nchoice < 7 else ((3,  self.nchoice/3) if  self.nchoice < 10 else (4, self.nchoice/4))
        self.nplot = (int(np.ceil(nplot[0])), int(np.ceil(nplot[1])))

        self.order = [(i, j) for i in range(self.nplot[0]) for j in  range(self.nplot[1])]

    def create_fig(self, size=(12, 7)):
        self.fig, self.ax = plt.subplots(self.nplot[0], self.nplot[1], figsize=size, sharex=True, sharey=True)

    def set_suptitle(self, title, size=18, weight="bold"):
        self.fig.suptitle(title, size=size, weight=weight)

    @staticmethod
    def get_fontdict(family="sans-serif", size=12, weight="normal"):
        return {
            "fontfamily": family,
            "fontsize": 12,
            "fontweight": weight,
        }

    def plot(self, x, y, title=None):
        if (self.nplot[0] != 1) and (self.nplot[1] != 1): 
            cur_ax = self.ax[self.order[self.count][0], self.order[self.count][1]]
            cur_ax.plot(x, y)
        else: 
            cur_ax = self.ax[self.count]
            cur_ax.plot(x, y)
        
        if (self.order[self.count][1] == 0) and (self.order[self.count][0] == self.nplot[0]-1): 
            pass
        elif self.order[self.count][1] == 0: 
            cur_ax.tick_params(axis=u'x', which=u'both',length=0)
        elif self.order[self.count][0] == self.nplot[0]-1: 
            cur_ax.tick_params(axis=u'y', which=u'both',length=0)
        else: 
            cur_ax.tick_params(axis=u'both', which=u'both',length=0)

        if not SUB_TITLED:
            if title != None: cur_ax.set_title(title)
            else: cur_ax.set_title(f"subplot_{self.count+1}")
            

        self.count += 1

    def add_labels(self, xlabel, ylabel, fontdict=None, padx=0, pady=12):
        self.fig.add_subplot(111, frameon=False)

        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)

        if fontdict == None: fontdict = self.get_fontdict()
        plt.xlabel(xlabel, fontdict=fontdict, labelpad=padx)
        plt.ylabel(ylabel, fontdict=fontdict, labelpad=pady)

    @staticmethod
    def convert_to_image():
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        return image

    # def save(self, fname):
    #     plt.tight_layout()
    #     plt.savefig(fname)
    #     self.reset()

    def reset(self):
        self.count = 0
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
    def __init__(self, train_dir, val_dir=None):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir

    def train_dataloader(self):
        return SpeakerVerificationDataLoader(
            SpeakerVerificationDataset(self.train_dir),
            speakers_per_batch=SPKR_PER_BATCH,
            utterances_per_speaker=UTTER_PER_SPKR,
            num_workers=NUM_WRKR,
        )
    
    def val_dataloader(self):
        return SpeakerVerificationDataLoader(
            SpeakerVerificationDataset(self.val_dir),
            speakers_per_batch=SPKR_PER_BATCH,
            utterances_per_speaker=UTTER_PER_SPKR,
            num_workers=NUM_WRKR,
        )

# ----------------------------------------------------------------------------------
class SpeakerEncoder(pl.LightningModule):
    def __init__(self, 
            hidden_layer_size=HL_SIZE, 
            n_layers=NUM_LAYERS, 
            learning_rate=LRATE
        ):
        super().__init__()   
        self.learning_rate = learning_rate 
        self.metrics = {
            "loss": [],
            "eer": [],
            "val_loss": [],
            "val_eer": []
        }

        self.metricPlotter = subPlotter()

        self.metricPlotter.create_fig()

        self.metricPlotter.set_suptitle("Training Loss by Step per Epoch", 21)

        self.epoch_idxs = np.random.choice(list(range(args.epochs)), self.metricPlotter.nchoice, False)

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
        embeds_loss = embeds.view((SPKR_PER_BATCH, UTTER_PER_SPKR, -1))

        loss, eer = self.loss(embeds_loss)
  
        self.logger.experiment.add_scalar(f"loss/{self.current_epoch}/{stage}", loss, batch_idx)
        self.logger.experiment.add_scalar(f"eer/{self.current_epoch}/{stage}", eer, batch_idx)

        # if self.current_epoch == 0: 
        #     self.logger.experiment.add_graph(SpeakerEncoder(), inputs)

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

        if self.current_epoch in self.epoch_idxs:
            self.metricPlotter.plot(range(len(loss)), loss)
        
        loss = np.mean(loss)
        eer = np.mean([x["eer"] for x in outputs])

        self.metrics["loss"].append(loss)
        self.logger.experiment.add_scalar("AvgLoss/train", loss, self.current_epoch)

        self.metrics["eer"].append(eer)
        self.logger.experiment.add_scalar("AvgEER/train", eer, self.current_epoch)    

    def validation_epoch_end(self, outputs):
        loss = [x["val_loss"].detach().item() for x in outputs]
        
        loss = np.mean(loss)
        eer = np.mean([x["val_eer"] for x in outputs])

        self.metrics["val_loss"].append(loss)
        self.logger.experiment.add_scalar("AvgLoss/val", loss, self.current_epoch)

        self.metrics["val_eer"].append(eer)
        self.logger.experiment.add_scalar("AvgEER/val", eer, self.current_epoch)

    def on_sanity_check_end(self):
        self.metrics = {}

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

        pl_module.metricPlotter.add_labels(
            "step (batch index)", "loss (GE2E loss)", 
            pl_module.metricPlotter.get_fontdict(size=14, weight=500)
        )

        # pl_module.metricPlotter.save("test.png")

        pl_module.logger.experiment.add_image(
            "plots\metrics\loss", 
            pl_module.metricPlotter.convert_to_image(), 
            pl_module.current_epoch
            )

        

        



# ----------------------------------------------------------------------
# def objective(trial):
#     nlayers = trial.suggest_int("n_layers", 1, 3)
#     hl_size = trial.suggest_int("hidden_layer_size", 64, 512, 64)
#     lrate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

#     print(nlayers, hl_size, lrate)

#     datamodule = SpeakerVerificationDataModule("../data/audio/train", "../data/audio/val")

#     model = SpeakerEncoder(hl_size, nlayers, lrate)
    
#     model_cb = SpeakerEncoderCallbacks()
#     pruning_cb= PyTorchLightningPruningCallback(trial, monitor="Loss/val")

#     tb_logger = pl.loggers.TensorBoardLogger("../lightning_logs")

#     trainer = pl.Trainer(
#         logger = tb_logger,
#         limit_train_batches = PER_TRAIN,
#         limit_val_batches = PER_VALID,
#         enable_checkpointing = False,
#         max_epochs = EPOCHS,
#         callbacks = [model_cb, pruning_cb],
#     )

#     hyperparameters = dict(hidden_layer_size=hl_size, n_layers=nlayers, learning_rate=lrate)
#     trainer.logger.log_hyperparams(hyperparameters)
#     trainer.fit(model, datamodule=datamodule)

#     print(trainer.callback_metrics)

#     # return trainer.callback_metrics["val_loss"].item()

# warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


if __name__=="__main__":


    # fig, ax = plt.subplots(2, 2)

    # x = np.linspace(0, 5, 100)

    # # Index 4 Axes arrays in 4 subplots within 1 Figure: 
    # ax[0, 0].plot(x, np.sin(x), 'g') #row=0, column=0
    # ax[1, 0].plot(range(100), 'b') #row=1, column=0
    # ax[0, 1].plot(x, np.cos(x), 'r') #row=0, column=1
    # ax[1, 1].plot(x, np.tan(x), 'k') #row=1, column=1

    # plt.savefig("test.png")


    # print()

    warnings.filterwarnings("ignore")

    # pruner = optuna.pruners.MedianPruner()

    # tb_callback = TensorBoardCallback("lightning_logs/", metric_name="val_loss")

    # study = optuna.create_study("Default Study", direction="maximize", pruner=pruner)

    # study.optimize(objective, n_trials=1)

    # print("Number of finished trials: {}".format(len(study.trials)))

    # print("Best trial:")
    # trial = study.best_trial

    # print(f"\tValue: {trial.value}")

    # print("\tParams: ")
    # for key, value in trial.params.items():
    #     print("\t\t{key}: {value}")


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


    ## --------------- FUNCTIONALITY TESTING ----------------------------------


    ## --- UTTERANCE --- 
    # root = "../data/Preprocessed/Audio/LibriSpeech/dev-clean/1272"

    # for upath in glob(root + "/*")[:5]:
    #     utter = Utterance(upath)
    #     print(utter)
    #     print(utter.frames_fpath)
    #     print(utter.get_frames().shape)
    #     arr = utter.random_partial(PAR_N_FRAMES)
    #     print(arr[0].shape, arr[1])
    #     print("---------------")

    ## --- SPEAKER ---
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # for spath in glob(root + "/*")[:5]: 
    #     spkr = Speaker(spath)
    #     print(spkr)
    #     for _, arr, _ in spkr.random_partial(4, PAR_N_FRAMES):
    #         print(arr.shape)

    ## SPEAKER BATCH
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # speakers = [Speaker(spath) for spath in glob(root+"/*")][:3]

    # test = SpeakerBatch(speakers, 4, PAR_N_FRAMES)

    # print(test)
    # print(test.data.shape)

    ## SPEAKER_VERIFICATION_DATASET
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # test = SpeakerVerificationDataset(root)

    # print(test)
    # print(test.__getitem__(1))
    # print(test.__getitem__(1).random_partial(4, 160))

    ## SPEAKER_VERIFICATION_DATALOADER
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # test = SpeakerVerificationDataLoader(SpeakerVerificationDataset(root), 4, 5)

    # print(test)
    # print(test.dataset)
    # print(test.dataset.__getitem__(1))
    # print(test.dataset.__getitem__(1).random_partial(4, 160))

    ## SPEAKER_VERIFICATION_DATAMODULE
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # test = SpeakerVerificationDataModule(root)

    # print(test) 
    # print(test.train_dataloader())
    # print(test.train_dataloader().dataset)
    # print(test.train_dataloader().dataset.__getitem__(5))
    # print(test.train_dataloader().dataset.__getitem__(5).random_partial(4, 160))