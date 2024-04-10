
import json
import os
import time
import torch
import cv2
from mobilesamv2 import sam_model_registry, SamPredictor
from typing import Any, Generator, List
import numpy as np
import urllib.request



class MobileSamBoxes:
    
    # encoder_path={'efficientvit_l2':efficient_vit_l2_path,
    #             'sam_vit_h':'weight/sam_vit_h.pt',}
        
    def __init__(self, img, boxesJsonPath=None, options = {}):
        self.img = img
        self.boxesJsonPath = boxesJsonPath
        self.download()
        
    def download(self):
        self.weights_path = 'weights'
        self.encoder_type = "efficientvit_l2"
        self.prompt_guided_path=os.path.join(self.weights_path,'Prompt_guided_Mask_Decoder.pt')
        self.efficient_vit_l2_path = os.path.join(self.weights_path,'l2.pt')
                
        # "https://mobile-sam.s3.eu-west-3.amazonaws.com/Prompt_guided_Mask_Decoder.pt"
        ptUrl = "https://mobile-sam.s3.eu-west-3.amazonaws.com/l2.pt"
        if not os.path.exists(self.efficient_vit_l2_path):
            print("download start")
            start = time.time()
            urllib.request.urlretrieve(ptUrl, self.efficient_vit_l2_path)
            print("------ download time: (s): %s" % round(time.time() - start, 2))

    
    def batch_iterator(self, batch_size: int, *args) -> Generator[List[Any], None, None]:
        assert len(args) > 0 and all(
            len(a) == len(args[0]) for a in args
        ), "Batched iteration must have inputs of al§l the same size."
        n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
        for b in range(n_batches):
            yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

    
    def create_model(self):
        PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](self.prompt_guided_path)
        mobilesamv2 = sam_model_registry['vit_h']()
        mobilesamv2.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
        mobilesamv2.mask_decoder=PromptGuidedDecoder['MaskDecoder']
        return mobilesamv2 
    
    def process(self, input_boxes = None):
        start = time.time()
        mobilesamv2= self.create_model()
        image_encoder=sam_model_registry[self.encoder_type](self.efficient_vit_l2_path)
        # image_encoder=sam_model_registry[self.encoder_type](self.encoder_path[self.encoder_type])
        mobilesamv2.image_encoder=image_encoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mobilesamv2.to(device=device)
        mobilesamv2.eval()
        predictor = SamPredictor(mobilesamv2)


        self.image = image = np.array(self.img)
        print("shape",image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        if input_boxes is None:
            input_boxes = self.readBoxesJson()

        input_boxes = np.array(input_boxes)
        input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes) #.cuda()
        sam_mask=[]
        image_embedding=predictor.features
        image_embedding=torch.repeat_interleave(image_embedding, 320, dim=0)
        prompt_embedding=mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding=torch.repeat_interleave(prompt_embedding, 320, dim=0)
        for (boxes,) in self.batch_iterator(320, input_boxes):
            with torch.no_grad():
                image_embedding=image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embedding=prompt_embedding[0:boxes.shape[0],:,:,:]
                sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,)
                low_res_masks, _ = mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                low_res_masks=predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)
                sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold)*1.0
                sam_mask.append(sam_mask_pre.squeeze(1))
        print("------ total time: (s): %s" % round(time.time() - start, 2))
        sam_mask=torch.cat(sam_mask)
        return sam_mask



    def readBoxesJson(self):
        path = self.boxesJsonPath 
        with open(path, 'r') as f:
            data = json.load(f)
        print("bb len",len(data))
        return data
        
        