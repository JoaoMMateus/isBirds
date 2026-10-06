from ddgs import DDGS
from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url
from pathlib import Path
from PIL import Image

searches = 'forest', 'birds', 'cats', 'dogs'
path = Path('imgs')

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
learn.fine_tune(3)

learn.export('bird_classifier.pkl')

learn_inf = load_learner('bird_classifier.pkl')
learn_inf.predict('img_b.jpg')
learn_inf.dls.vocab