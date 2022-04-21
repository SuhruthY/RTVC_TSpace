from utils import *

logger = logging.getLogger("Functionality Tester")

logger.addHandler(file_handler)

log(logger, "Project Name: RTVC - Model Name: Encoder - Stage Name: Functionality Testing")
# --- UTTERANCE --- 
log(logger, "Testing - Class: Utterance")

for upath in glob(ROOT + "/1272/*")[:5]:
    utter = Utterance(upath)

    log(logger, f"Instance: {utter}")
    log(logger, f"File Path: {utter.frames_fpath}")

    arr = utter.get_frames()
    log(logger, f"Raw Frames: {lst2str(arr[0][:5])} - Shape: {arr.shape}")

    arr = utter.random_partial(PAR_N_FRAMES)

    log(logger, f"Taking random partial from index {arr[1][0]} to index {arr[1][1]}")
    log(logger, f"Random Partials: {lst2str(arr[0][0][:5])} - Shape: {arr[0].shape}")

    break

# --- SPEAKER ---
log(logger, "Testing - Class: Speaker")

for spath in glob(ROOT + "/*")[:5]: 
    spkr = Speaker(spath)
    log(logger, f"Instance: {spkr}")
    
    for _, arr, _ in spkr.random_partial(4, PAR_N_FRAMES):
        log(logger, f"Random Partials: {lst2str(arr[0][:5])} -  Shape: {arr.shape}")

        break
               
    break
    

# SPEAKER BATCH
log(logger, "Testing - Class: Speaker Batch")

speakers = [Speaker(spath) for spath in glob(ROOT+"/*")][:3]

batch = SpeakerBatch(speakers, 4, PAR_N_FRAMES)

log(logger, f"Instance: {batch} - Shape: {batch.data.shape}")

# SPEAKER_VERIFICATION_DATASET
log(logger, "Testing - Class: Speaker Verification Dataset")

ds = SpeakerVerificationDataset(ROOT)

log(logger, f"Instance: {ds}")

arr = ds.__getitem__(1)
log(logger, f"Item: {arr}")

_, arr, _ = arr.random_partial(4, 160)[0]
log(logger, f"Random Partials: {lst2str(arr[0][:5])} - Shape: {arr.shape}")



# SPEAKER_VERIFICATION_DATALOADER
log(logger, "Testing - Class: Speaker Verification Data Loader")

dl = SpeakerVerificationDataLoader(SpeakerVerificationDataset(ROOT), 4, 5)

log(logger, f"Instance: {dl}")
log(logger, f"Item: {dl.dataset} - Sub Item: {dl.dataset.__getitem__(1)}")

_, arr, _ = dl.dataset.__getitem__(1).random_partial(4, 160)[0]
log(logger, f"Random Partials: {lst2str(arr[0][:5])} - Shape: {arr.shape}")



# SPEAKER_VERIFICATION_DATAMODULE
log(logger, "Testing - Class: Speaker Verification Data Module")

dm = SpeakerVerificationDataModule(ROOT)

log(logger, f"Instance: {dm}")

tdl = dm.train_dataloader()
log(logger, f"Item: {tdl} -  Sub Item: {tdl.dataset} - Sub Item 2: {tdl.dataset.__getitem__(1)}")

_, arr, _ = tdl.dataset.__getitem__(1).random_partial(4, 160)[0]
log(logger, f"Random Partials: {lst2str(arr[0][:5])} - Shape: {arr.shape}")

log(logger, "-"*100)