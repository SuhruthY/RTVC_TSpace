import struct
import librosa
import webrtcvad
import numpy as np

import logging

from glob import glob
from scipy.ndimage.morphology import binary_dilation

AUDIO_RE = "[fw][la][av]*"

TARG_FS = 16000 
TARG_dbFS = -30
INT16_MAX = (2**15) - 1

VAD_WNDW_LEN = 30  
VAD_WNDW_AVG_WIDTH = 8
VAD_MAX_SILENCE_LEN = 6

MEL_WNDW_LEN = 25  
MEL_WNDW_STP = 10    
MEL_N_CHANNELS = 40 

PAR_N_FRAMES = 160 


def normalize_volume(wav, target_dBFS=TARG_dbFS, increase_only=False, decrease_only=False):
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))

def moving_average(array, width):
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    ret = np.cumsum(array_padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width

def trim_long_silences(wav, sampling_rate=TARG_FS):
    samples_per_window = (VAD_WNDW_LEN * sampling_rate) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * INT16_MAX)).astype(np.int16))

    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2], sample_rate=sampling_rate))
    audio_mask = moving_average(voice_flags, VAD_WNDW_AVG_WIDTH)
    audio_mask = np.round(audio_mask).astype(np.bool_)
    audio_mask = binary_dilation(audio_mask, np.ones(VAD_MAX_SILENCE_LEN + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    return wav[audio_mask == True]


def load_audio(file_path):
    arr, _ = librosa.load(file_path) 
    return arr

def preprocess_data(arr):
        return trim_long_silences(normalize_volume(arr))

def generate_frames(preprocessed_wav, sampling_rate=TARG_FS):
    frames = librosa.feature.melspectrogram(
    preprocessed_wav,
    sampling_rate,
    n_fft=int(sampling_rate * MEL_WNDW_LEN / 1000),
    hop_length=int(sampling_rate * MEL_WNDW_STP / 1000),
    n_mels=MEL_N_CHANNELS
    )
    return frames.astype(np.float32).T

def get_metadata(file_path):
    meta = am.load(file_path)
    return json.dumps({
        "full_path": meta["filepath"],
        "file_size": meta["filesize"],
        "bit_depth": meta["streaminfo"]["bit_depth"],
        "bit_rate": meta["streaminfo"]["bitrate"],
        "duration": meta["streaminfo"]["duration"]
    }, indent=4)


import os
import time
import datetime as dt
import audio_metadata as am

import json


today = dt.datetime.today()
LOG_FNAME = f"{today.day:02d}-{today.month:02d}-{today.year}.log"
LOG_FORMAT = '%(asctime)s: %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

formatter = logging.Formatter(LOG_FORMAT)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

logger = logging.getLogger("Preprocess Logger")

file_handler = logging.FileHandler(f"./log/{LOG_FNAME}")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


logger.info(f"Project Name: RTVC - Model Name: Encoder - Stage Name: Preprocessing")
time.sleep(1)

root = "../data"
op_dir = "../data/PreProcessed/Audio"

# DSET_NAMES = ["LibriSpeech", "LibriTTS"]

# DSET_NAMES = ["LibriSpeech"]

DSET_NAMES = ["LibriTTS"]

# SPLT_NAMES = ["dev-clean", "test-clean", "train-clean-100"]

SPLT_NAMES = ["train-clean-100"]

try:
    for dname in DSET_NAMES:
        for split in SPLT_NAMES: 
            logger.info(f"Currently working with {dname}_Dataset in {split} partiiton")
            for spath in glob(f"{root}/{dname}/{split}/*[0-9]"):
                s_id= spath.split("\\")[-1]
                logger.info(f"Opening Speaker ID: {s_id}")
                for fpath in glob(f"{spath}/*/*{AUDIO_RE}"):
                    f_id, ftype = fpath.split("\\")[-1].split(".")

                    out_path = op_dir + f"/{dname}/{split}/{s_id}"

                    if not os.path.exists(out_path): os.makedirs(out_path)

                    out_fpath = out_path + f"/{f_id}.npy"

                    if os.path.exists(out_fpath):
                        logger.warning(f"A file with file name {f_id}.npy already exists")
                        logger.info("Default: Aborting iteration...")
                        continue    

                    logger.info(f"Loading File ID: {f_id}")
                    time.sleep(1)

                    logger.info("Audio Stats:")
                    logger.info(f"{get_metadata(fpath)}")
                    time.sleep(1)

                    wav = load_audio(fpath)

                    logger.info(f"Loaded an audio file with shape - {wav.shape}")

                    wav = preprocess_data(wav)
                    if len(wav) == 0: 
                        logger.warning("audio_len = 0: No Information in audio file")
                        logger.info(f"Ignoring File_{f_id} and Continue...")
                        logger.info("-"*21)
                        continue
                        
                    logger.info("Generating frames...Expecting shape (x, 40):x < 160")    

                    frames = generate_frames(wav)
                    if len(frames) < PAR_N_FRAMES: 
                        logger.warning(f"generated frames length not meets requirements ({len(frames)})")
                        logger.info(f"Ignoring File_{f_id} and Continue...")
                        logger.info("-"*21)
                        continue    

                    logger.info(f"Saving frames of size {frames.shape}")
                    
                    np.save(out_fpath, frames)

                    # np.save(out_fpath, frames.T.astype(np.float32)) #should useed this

                    logger.info(f"File {f_id}.{ftype} is preprocessed and saved as {f_id}.npy")
                    logger.info("-"*21)
except Exception as ex:
    logger.error(f"An Exception occurred - {ex}") 
    logger.info("Aborting Process...")

logger.info( "-"*21 + "Process Finished Successfully!")
        
