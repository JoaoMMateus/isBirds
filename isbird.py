from ddgs import DDGS
from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url
from pathlib import Path
from PIL import Image
import ipywidgets as widgets

#image = Path(__file__).parent / 'img_b.jpg'
#image = Path(__file__).parent / 'img_f.jpg'

# Load the saved learner
learn = load_learner('bird_classifier.pkl')

# Use it for prediction
#is_bird, _, probs = learn.predict(image)
#print(f"This is a: {is_bird}.")
#print(f"Probability it's a bird: {probs[0]:.4f}")

btn_upload = widgets.FileUpload()
btn_upload = SimpleNamespace(data = ['img_f.jpg'])
img = PILImage.create(btn_upload.data[-1])

# out_pl = widgets.Output()
# out_pl.clear_output()
# with out_pl: display(img.to_thumb(128,128))
# out_pl

pred, pred_idx, probs = learn.predict(img)

print(f"This is a: {pred}.")
print(f"This is: {pred_idx}.")
print(f"Probability it's a bird: {probs[0]:.4f}")

lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred

btn_run = widgets.Button(description='Classify')
btn_run

def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)