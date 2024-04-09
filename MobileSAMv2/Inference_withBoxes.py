import numpy as np
from MobileSamBoxes import MobileSamBoxes
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
    sam = MobileSamBoxes("./app/assets/picture1.jpg",
                         "./notebooks/boxes.json")
    anns = sam.process()
    plt.figure(figsize=(10,10))
    background=np.ones_like(sam.image)*255
    plt.imshow(background)
    show_anns(anns)
    plt.axis('off')
    plt.show() 
    plt.savefig("{}".format("./out/result.jpg"), bbox_inches='tight', pad_inches = 0.0) 
    