import time
import cv2
import joblib
import os
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torch

modelPath = "/Users/eliaweiss/Documents/doc2txt/sihach/model"
# Load the model
model = joblib.load(os.path.join(modelPath,"my_svm_model.pkl"))

# Load the scaler
scaler = joblib.load(os.path.join(modelPath,"my_scaler.pkl"))

sam_checkpoint = "./weights/mobile_sam.pt"
model_type = "vit_t"
device = "cuda" if torch.cuda.is_available() else "cpu"


mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

def encodeEmbSAM(image1):
  image1 = cv2.imread(image1)
  predictor.set_image(image1)
  img1=predictor.features
  img1 = img1.view(1, -1) 
  return img1
def predict_category(model, scaler, image):
    """
    Predicts the category of a new image.

    Args:
        model: Trained SVM classifier.
        scaler: StandardScaler used for preprocessing.
        image: Path to the new image.

    Returns:
        predicted_category: Category assigned to the image.
    """
    startTime = time.time()
    embedding = encodeEmbSAM(image)
    embedding = embedding.squeeze()
    print("------ encodeEmbSAM time: (s): %s" % round(time.time() - startTime, 2))

    startTime = time.time()    
    embedding_scaled = scaler.transform([embedding])
    predicted_category = model.predict(embedding_scaled)[0]
    print("------ predict time: (s): %s" % round(time.time() - startTime, 2))
   
    return predicted_category
new_image_paths = [
    "/Users/eliaweiss/Documents/doc2txt/sihach/Invoices/1/img1/1.jpg",
    "/Users/eliaweiss/Documents/doc2txt/sihach/Invoices/2/img2/1.jpg",
    "/Users/eliaweiss/Documents/doc2txt/sihach/Invoices/3/img3/1.jpg",
]
for new_image_path in new_image_paths:
    predicted_category = predict_category(model, scaler, new_image_path)
    print(f"Predicted category for {new_image_path}: {predicted_category}")