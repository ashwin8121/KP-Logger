from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import PIL.Image as pi  
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from timeit import default_timer
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classes = {0: '0', 
1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}

model = load_model("model.h5")

class Data(BaseModel):
    image: str

@app.get("/")
def index():
    return {"Message": "Hello, use the /get_file Post method to decode captcha"}


@app.post("/get_file")
def get_file(json: Data):
    try:
        img = urlretrieve(json.image, filename="out.png")
        out = predict(img[0])
        return {"message": out}
    except Exception as e:
        return {"message": "retry", "error": str(e)}



def predict(img_path):
    image = pi.open(img_path).convert("L")
    img = np.array(image)
    plt.imsave(r"tempimg.jpg", img, cmap="binary")
    img = plt.imread(r"tempimg.jpg")
    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    pts = [[(7, 0), (27, 40)], [(27, 0), (47, 40)], [(47, 0), (67, 40)], [(67, 0), (87, 40)], [(87, 0), (107, 40)], [(107, 0), [127, 40]]]
    imgs = []
    for (x1, _), (x2, _) in pts:
        i = img[:, x1:x2]
        imgs.append(i)
    imgs = np.array(imgs).reshape(-1, 40, 20, 1)
    output = model.predict(imgs, verbose=0)
    cp = ""
    for o in output:
        cp += classes[np.argmax(o)]
    os.remove("out.png")
    os.remove("tempimg.jpg")
    return { "text": cp}


port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
