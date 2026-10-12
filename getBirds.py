from ddgs import DDGS
from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url
import gradio as gr
from pathlib import Path

import gradio as gr
from fastai.vision.all import load_learner, PILImage

# Set to True or False to skip or not the downloading of images
skip_download = True

searches = 'forest', 'birds', 'cats', 'dogs'
path = Path('imgs')

if not skip_download:
    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        urls = DDGS().images(
            query=f"{o} photos",
            region="us-en",
            safesearch="on",
            max_results=5,
            page=1,
            backend="auto",
            size=None,
            color="color",
            type_image="photo",
            layout="square",
            license_image="Public",
        )
        for u in urls:
            print(u['title'])
            download_url(u['image'], dest, show_progress=False)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=2, num_workers=0)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows=1)

learn.export('bird_classifier.pkl')