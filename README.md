python 3.9

# Faster Segment Anything (MobileSAM) and Everything (MobileSAMv2)
:pushpin: MobileSAMv2, available at [ResearchGate](https://www.researchgate.net/publication/376579294_MobileSAMv2_Faster_Segment_Anything_to_Everything) and [arXiv](https://arxiv.org/abs/2312.09579.pdf), replaces the grid-search prompt sampling in SAM with object-aware prompt sampling for faster **segment everything(SegEvery)**.

:pushpin: MobileSAM, available at [ResearchGate](https://www.researchgate.net/publication/371851844_Faster_Segment_Anything_Towards_Lightweight_SAM_for_Mobile_Applications) and [arXiv](https://arxiv.org/pdf/2306.14289.pdf), replaces the heavyweight image encoder in SAM with a lightweight image encoder for faster **segment anything(SegAny)**. 








## Installation

The code requires `python>=3.9`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Mobile Segment Anything:

```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

or clone the repository locally and install with

```
git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM; pip install -e .
```

## Demo

Once installed MobileSAM, you can run demo on your local PC or check out our [HuggingFace Demo](https://huggingface.co/spaces/dhkim2810/MobileSAM).

It requires latest version of [gradio](https://gradio.app).

```
cd app
python app.py
```

## <a name="GettingStarted"></a>Getting Started
The MobileSAM can be loaded in the following ways:

```
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from mobile_sam import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(<your_image>)
```
## <a name="GettingStarted"></a>Getting Started (MobileSAMv2)
Download the model weights from the [checkpoints](https://drive.google.com/file/d/1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE/view?usp=sharing).

After downloading the model weights, faster SegEvery with MobileSAMv2 can be simply used as follows:
```
cd MobileSAMv2
bash ./experiments/mobilesamv2.sh
```
## ONNX Export
**MobileSAM** now supports ONNX export. Export the model with

```
python scripts/export_onnx_model.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./mobile_sam.onnx
```

Also check the [example notebook](https://github.com/ChaoningZhang/MobileSAM/blob/master/notebooks/onnx_model_example.ipynb) to follow detailed steps.
We recommend to use `onnx==1.12.0` and `onnxruntime==1.13.1` which is tested.




## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University))

<details>
<summary>
<a href="https://github.com/facebookresearch/segment-anything">SAM</a> (Segment Anything) [<b>bib</b>]
</summary>

</details>



<details>
<summary>
<a href="https://github.com/microsoft/Cream/tree/main/TinyViT">TinyViT</a> (TinyViT: Fast Pretraining Distillation for Small Vision Transformers) [<b>bib</b>]
</summary>
