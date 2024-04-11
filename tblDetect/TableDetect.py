import torchvision.transforms as T
import torch
torch.set_grad_enabled(False);

from transformers import AutoModelForObjectDetection

# from transformers import TableTransformerForObjectDetection


class TableDetect:
    def __init__(self):
        # standard PyTorch mean-std input image normalization
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")        
        # self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.model.eval();

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b        
        
      
    
    def detectTables(self,im):
        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(im).unsqueeze(0)

        # propagate through the model
        outputs = self.model(img)
        probas = outputs['logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.1
        probas = probas[keep]

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)   
        
        return probas, bboxes_scaled 