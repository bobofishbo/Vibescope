import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),         
    transforms.Normalize(         
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  
    input_tensor = transform(image).unsqueeze(0)  
    return input_tensor, image

model = models.resnet50(pretrained=True)
model.eval() 

image_path = "/Users/andrew131/Desktop/CVCV/Screenshot 2024-12-24 at 08.57.01.png" 
input_tensor, original_image = load_image(image_path)

with torch.no_grad():
    features = model(input_tensor)

age_prediction = np.mean(features.numpy()) * 10
print(f"Predicted Age: {age_prediction:.2f}")

personality_scores = features.numpy()[0][:4]
personality_labels = ['Introvert/Extrovert', 'Sensing/Intuition', 'Thinking/Feeling', 'Judging/Perceiving']
personality_map = {label: score for label, score in zip(personality_labels, personality_scores)}

print("\nPredicted Personality:")
for trait, score in personality_map.items():
    print(f"{trait}: {score:.2f}")

plt.imshow(original_image)
plt.axis("off")
plt.title(f"Predicted Age: {age_prediction:.2f}")
plt.show()
