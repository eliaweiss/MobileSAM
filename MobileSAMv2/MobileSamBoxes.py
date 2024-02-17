
import json
import time
import torch
import cv2
from mobilesamv2 import sam_model_registry, SamPredictor
from typing import Any, Generator, List
import numpy as np



class MobileSamBoxes:
    
    encoder_path={'efficientvit_l2':'MobileSAMv2/weight/l2.pt',
                'tiny_vit':'MobileSAMv2/weight/mobile_sam.pt',
                'sam_vit_h':'MobileSAMv2/weight/sam_vit_h.pt',}
        
    def __init__(self, imagePath, boxesJsonPath, options = {}):
        self.imagePath = imagePath
        self.boxesJsonPath = boxesJsonPath
        self.encoder_type = "efficientvit_l2"
        self.prompt_guided_path='MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'
        
        
        

    
    def batch_iterator(self, batch_size: int, *args) -> Generator[List[Any], None, None]:
        assert len(args) > 0 and all(
            len(a) == len(args[0]) for a in args
        ), "Batched iteration must have inputs of all the same size."
        n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
        for b in range(n_batches):
            yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

    
    def create_model(self):
        PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](self.prompt_guided_path)
        mobilesamv2 = sam_model_registry['vit_h']()
        mobilesamv2.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
        mobilesamv2.mask_decoder=PromptGuidedDecoder['MaskDecoder']
        return mobilesamv2 
    
    def process(self):
        img_fullpath=  self.imagePath
        start = time.time()
        mobilesamv2= self.create_model()
        image_encoder=sam_model_registry[self.encoder_type](self.encoder_path[self.encoder_type])
        mobilesamv2.image_encoder=image_encoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mobilesamv2.to(device=device)
        mobilesamv2.eval()
        predictor = SamPredictor(mobilesamv2)


        print(">>>",img_fullpath)
        self.image = image = cv2.imread(img_fullpath)
        print("shape",image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_boxes1 = self.readBoxesJson()

        input_boxes = np.array(input_boxes1)
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
        # annotation = sam_mask
        # areas = torch.sum(annotation, dim=(1, 2))
        # sorted_indices = torch.argsort(areas, descending=True)
        # anns = annotation[sorted_indices]



    def readBoxesJson(self):
        path = self.boxesJsonPath # "/Users/eliaweiss/ai/MobileSAM/notebooks/boxes.json"
        with open(path, 'r') as f:
            data = json.load(f)
        print("bb len",len(data))
        return data
        