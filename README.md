# tdt17

The files associated with our intial unet model are unet.py and trainer_unet.py.

For the largest part of our project we used trainer_pretrained.py, which contains our pre-trained unet model.

For the attentionUnet runs, we used attentionunet.py alongside the plain trainer.py file, which we also used for some random testing with existing models, like importing unet from monai.

Finally, our datamodule.py file contains any preprocessing we did including data augmentations. The config file was also used to maintain certain variables across different models. The config file is taken from the class tdt4265's tutorial. We also used the setup for wandb from that class.