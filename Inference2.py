import numpy as np
from tblDetect.MobileSamBoxes import MobileSamBoxes
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)


if __name__ == "__main__":
    sam = MobileSamBoxes("/Users/eliaweiss/work/ocrPlus/result/678e7ef0c034689d124d13df47c58695.jpg",
                         "/Users/eliaweiss/ai/MobileSAM/notebooks/boxes.json")
    anns = sam.process()
    plt.figure(figsize=(10,10))
    background=np.ones_like(sam.image)*255
    plt.imshow(background)
    show_anns(anns)
    plt.axis('off')
    plt.show() 