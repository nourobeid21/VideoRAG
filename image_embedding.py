from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import json

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

assocs = json.load(open("data/frames/associations.json"))
img_embs = []

for a in assocs:
    img = Image.open(f"data/frames/frame_{a['frame_id']}.jpg").convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    emb = clip.get_image_features(**inputs).detach().cpu().numpy()[0]
    img_embs.append(emb)

np.save("data/embeddings/img_embs.npy", np.vstack(img_embs))
