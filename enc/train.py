from utils import *

datamodule = SpeakerVerificationDataModule(f"data/audio/train", f"data/audio/val")

model = SpeakerEncoder()   
model_cb = SpeakerEncoderCallbacks()

tb_logger = pl.loggers.TensorBoardLogger("../lightning_logs", args.version_name)

CHECK_MODEL = True if args.version_name!="default" else False
GPUS = torch.cuda.device_count() if torch.cuda.is_available() else None

print(CHECK_MODEL, GPUS)

trainer = pl.Trainer(callbacks=[model_cb],
                    logger=tb_logger,
                    fast_dev_run=args.fast_run, 
                    max_epochs=args.epochs,
                    enable_checkpointing=check_model,
                    gpus=GPUS,
                    limit_train_batches=args.per_train,
                    limit_val_batches=args.per_val,
                    )

# trainer.fit(model, datamodule=datamodule)