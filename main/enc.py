import warnings
warnings.filterwarnings("ignore")

import os
import time
import random
import numpy as np

from glob import glob

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq

PAR_N_FRAMES = 160
MODEL_EM_size = 256
MEL_N_CHANNELS = 40


WORKERS = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()


model_hidden_size = 256
model_num_layers = 3
learning_rate_init = 1e-4

speakers_per_batch = 8 # 64
utterances_per_speaker = 5 # 10

epochs = 20
version="v00"

limit_batch = 100
save_every = 10

# save_model = False

save_model = True

# base_dir = "../data/audio"
# model_dir = "../data/model"

base_dir = "../input/real-time-voice-cloning"
model_dir = "./model"

if not os.path.exists(model_dir): os.makedirs(model_dir)

## UTILS   
import time

def loadbar(iteration, total, prefix="", suffix="", decimal=0, 
            length=100, fill="=", extras=""):
    per_val = iteration*100/float(total)
    
    percent = ("{0:." + str(decimal) + "f}").format(per_val)
    
    cur_percent = ( ' ' * (len(str(total))-len(str(round(per_val)))) + percent)
    
    filledLen = int(length * iteration//total)
    if per_val == 100:
        bar = fill * filledLen + "." * (length - filledLen)
    else:
        bar = fill * filledLen + ">" + "." * (length - filledLen - 1)
    print(f"\r{prefix} [{bar}] {cur_percent}% {suffix}", end="\r")
    if iteration == total: 
        print(f"\r{prefix} [{bar}] {cur_percent}% {suffix} {extras}", end="\n")
        
    time.sleep(0.1)

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
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=WORKERS, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
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

## MODEL
class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        
        self.lstm = nn.LSTM(input_size=MEL_N_CHANNELS,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=MODEL_EM_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self):
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        return embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        
    
    def similarity_matrix(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        return sim_matrix * self.similarity_weight + self.similarity_bias
    
    def loss(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer

def sync(device: torch.device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    

def train(
    epoch=0,
    run_id=version,  
    ):

    train_dl = SpeakerVerificationDataLoader(
        SpeakerVerificationDataset(base_dir + "/train"),
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=WORKERS
    )
    
    valid_dl = SpeakerVerificationDataLoader(
        SpeakerVerificationDataset(base_dir + "/val"),
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=WORKERS
    )
    
    bsize = len(loader) if limit_batch == -1 else limit_batch    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    step = 0
    
    if save_model:
        state_fpath = f"{model_dir}/encoder_{run_id}.pt"
        if os.path.isfile(state_fpath):
            checkpoint = torch.load(state_fpath)
            step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init   

    model.train()

    res = {
        "loss":0, 
        "eer":0,
        "val_loss": 0,
        "val_eer": 0,
    }
    
    i = 0
    cur_epoch = ( '0' * (len(str(epochs))-len(str(epoch))) + str(epoch))
    loadbar(i, bsize, f"Epoch: [{cur_epoch}/{epochs}]  {i}/{bsize}", length=50)
    for idx, speaker_batch in enumerate(train_dl):
        
        step += 1
 
        if idx > bsize-1: break

        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)

        embeds = model(inputs)
        sync(device)

        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        
        sync(loss_device)

        # Backward pass
        model.zero_grad()
        loss.backward()

        model.do_gradient_ops()
        optimizer.step()

        res["loss"] += loss.detach().item()
        res["eer"] += eer
        
        cur_batch = ( '0' * (len(str(bsize))-len(str(i+1))) + str(i+1))
        p = f"Epoch [{cur_epoch}/{epochs}] {cur_batch}/{bsize}"
        
        if i+1 == bsize:
            cur_batch = ( '0' * (len(str(bsize))-len(str(i))) + str(i))
            p = f"Epoch [{cur_epoch}/{epochs}] {cur_batch}/{bsize}"
            s = f"- loss:{res['loss']/bsize:0.4f} - eer:{res['eer']/bsize:0.4f}"
            loadbar(i-1, bsize, p, s, length=50)
            for idx, speaker_batch in enumerate(valid_dl):

                if idx > bsize-1: break

                # Forward pass
                inputs = torch.from_numpy(speaker_batch.data).to(device)
                sync(device)

                embeds = model(inputs)
                sync(device)

                embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
                loss, eer = model.loss(embeds_loss)

                sync(loss_device)
                
                res["val_loss"] += loss
                res["val_eer"] += eer
                                
                e = f" - val_loss:{loss:0.4f} - val_eer:{eer:0.4f}"
                loadbar(i-1, bsize, p, s+e, length=50)
                  
            cur_batch = ( '0' * (len(str(bsize))-len(str(i+1))) + str(i+1))
            p = f"Epoch [{cur_epoch}/{epochs}] {cur_batch}/{bsize}"
#             s = f"- loss:{res['loss']/bsize:0.4f} - eer:{res['eer']/bsize:0.4f}"
            e = f" - val_loss:{res['val_loss']/bsize:0.4f} - val_eer:{res['val_eer']/bsize:0.4f}"
#             loadbar(i+1, bsize, p, s, extras=e, length=50)
            loadbar(i+1, bsize, p, s+e, length=50)
        else:
            s = f"- loss:{loss:0.4f} - eer:{eer:0.4f}"
            loadbar(i+1, bsize, p, s, length=50)
            
        i+=1
       

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0 and save_model:
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)

for epoch in range(epochs): train(epoch)

## Testing Utils 
# base_dir = "../data/audio/train"

# utter = Utterance(glob(base_dir + "/1272/*")[0])
# arr = utter.get_frames()
# print(arr.shape)
# arr, _ = utter.random_partial(PAR_N_FRAMES)
# print(arr.shape)

# spkr = Speaker(glob(base_dir + "/*")[0])
# _, arr, _ = spkr.random_partial(4, PAR_N_FRAMES)[0] 
# print(arr.shape)

# speakers = [Speaker(spath) for spath in glob(base_dir+"/*")][:3]
# batch = SpeakerBatch(speakers, 4, PAR_N_FRAMES)
# print(batch.data.shape)

# dl = SpeakerVerificationDataLoader(SpeakerVerificationDataset(base_dir), 4, 5)
# _, arr, _ = dl.dataset.__getitem__(1).random_partial(4, 160)[0]
# print(arr.shape)

# for epoch in range(epochs): 
#     i = 0
#     print(f"Epoch: [{epoch}/{epochs}]")
#     loadbar(i, bsize, f"  {i}/{bsize}", length=50)
#     loss, eer = train()
#     print(f"Epoch:{epoch}/{epochs} - Loss:{loss:0.4f} - EER:{eer:0.4f}")

## Testing LOADBAR
# epochs=10
# bsize = 125


# for epoch in range(epochs):
#     i = 0
#     print(f"Epoch: [{epoch}/{epochs}]")
#     loadbar(i, bsize, f"  {i}/{bsize}", length=50)
#     for batch in range(bsize):
#         p = f"{( '0' * (len(str(bsize))-len(str(i+1))) + str(i+1))}/{bsize}"
#         loadbar(i+1, bsize, p, "- Hello", length=50)
#         i+=1