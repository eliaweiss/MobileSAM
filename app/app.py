import os
import time

import gradio as gr
import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import ImageDraw
from utils.tools import box_prompt, format_results, point_prompt
from utils.tools_gradio import fast_process

# Most of our demo code is from [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM). Huge thanks for AN-619.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
sam_checkpoint = "../weights/mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)

# Description
title = "<center><strong><font size='8'>Faster Segment Anything(MobileSAM)<font></strong></center>"

description_e = """This is a demo of [Faster Segment Anything(MobileSAM) Model](https://github.com/ChaoningZhang/MobileSAM).

                   We will provide box mode soon. 

                   Enjoy!
                
              """

description_p = """ # Instructions for point mode

                0. Restart by click the Restart button
                1. Select a point with Add Mask for the foreground (Must)
                2. Select a point with Remove Area for the background (Optional)
                3. Click the Start Segmenting.

              """

examples = [
    ["assets/picture0.jpg"],
    # ["assets/picture3.jpg"],
    # ["assets/picture4.jpg"],
    # ["assets/picture5.jpg"],
    # ["assets/picture6.jpg"],
    # ["assets/picture1.jpg"],
    # ["assets/picture2.jpg"],
]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    points_per_side=5,
    stability_score_thresh=0.1,
    pred_iou_thresh=0.9,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    print("points_per_side",points_per_side)
    print("stability_score_thresh",stability_score_thresh)
    print("input_size",input_size)
    startTime = time.time()
    # global mask_generator
    global mobile_sam
    mask_generator = SamAutomaticMaskGenerator(
        model=mobile_sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        # crop_n_layers=1,
        # crop_n_points_downscale_factor=2,
        # min_mask_region_area=100,  # Requires open-cv to run post-processing
)    

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    
    print("------ total time: (s): %s" % round(time.time() - startTime, 2))

    return fig


def segment_with_points(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label
    startTime = time.time()

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    
    # print("global_points",global_points)

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_point_label = np.array(global_point_label)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, image

    # print(scaled_points, scaled_points is not None)
    # print(scaled_point_label, scaled_point_label is not None)

    nd_image = np.array(image)
    predictor.set_image(nd_image)
    masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=True,
    )

    results = format_results(masks, scores, logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array([annotations])

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    print("------ total time: (s): %s" % round(time.time() - startTime, 2))

    global_points = []
    global_point_label = []
    # return fig, None
    return fig, image


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == "Add Mask" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    print(x, y, label == "Add Mask")

    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image


cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
cond_img_p = gr.Image(label="Input with points", value=default_example[0], type="pil")

segm_img_e = gr.Image(label="Segmented Image", interactive=False, type="pil")
segm_img_p = gr.Image(
    label="Segmented Image with points", interactive=False, type="pil"
)

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(
    minimum=512,
    maximum=1024,
    value=1024,
    step=64,
    label="Input_size",
    info="Our model was trained on a size of 1024",
)

points_per_side_slider = gr.components.Slider(
    minimum=1,
    maximum=32,
    value=5,
    step=1,
    label="points_per_side",
    info="points_per_side",
)
stability_score_thresh_slider = gr.components.Slider(
    minimum=0.01,
    maximum=1,
    value=0.1,
    step=0.01,
    label="stability_score_thresh",
    info="stability_score_thresh",
)
pred_iou_thresh_slider = gr.components.Slider(
    minimum=0.01,
    maximum=1,
    value=0.92,
    step=0.01,
    label="pred_iou_thresh",
    info="pred_iou_thresh",
)

with gr.Blocks(css=css, title="Faster Segment Anything(MobileSAM)") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["Add Mask", "Remove Area"],
                        value="Add Mask",
                    )

                    with gr.Column():
                        segment_btn_p = gr.Button(
                            "Start segmenting!", variant="primary"
                        )
                        clear_btn_p = gr.Button("Restart", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_p],
                    # outputs=segm_img_p,
                    # fn=segment_with_points,
                    # cache_examples=True,
                    examples_per_page=4,
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)


    with gr.Tab("Everything mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_e.render()
    
            with gr.Column(scale=1):
                segm_img_e.render()
    
        # Submit & Clear
        with gr.Row():
            with gr.Column():
                input_size_slider.render()
                points_per_side_slider.render()
                stability_score_thresh_slider.render()
                pred_iou_thresh_slider.render()
                with gr.Row():
                    contour_check = gr.Checkbox(
                        value=True,
                        label="withContours",
                        info="draw the edges of the masks",
                    )
    
                    with gr.Column():
                        segment_btn_e = gr.Button(
                            "Segment Everything", variant="primary"
                        )
                        clear_btn_e = gr.Button("Clear", variant="secondary")
    
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_e],
                    outputs=segm_img_e,
                    fn=segment_everything,
                    cache_examples=False,
                    examples_per_page=4,
                )
    
            with gr.Column():
                with gr.Accordion("Advanced options", open=False):
                    # text_box = gr.Textbox(label="text prompt")
                    with gr.Row():
                        mor_check = gr.Checkbox(
                            value=False,
                            label="better_visual_quality",
                            info="better quality using morphologyEx",
                        )
                        with gr.Column():
                            retina_check = gr.Checkbox(
                                value=True,
                                label="use_retina",
                                info="draw high-resolution segmentation masks",
                            )
                # Description
                gr.Markdown(description_e)
    
    
    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)

    segment_btn_e.click(
        segment_everything,
        inputs=[
            cond_img_e,
            input_size_slider,
            points_per_side_slider,
            stability_score_thresh_slider,
            pred_iou_thresh_slider,
            mor_check,
            contour_check,
            retina_check,
        ],
        outputs=segm_img_e,
    )

    segment_btn_p.click(
        segment_with_points, inputs=[cond_img_p], outputs=[segm_img_p, cond_img_p]
    )

    def clear():
        return None, None

    def clear_text():
        return None, None, None

    # clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])

demo.queue()
demo.launch()
