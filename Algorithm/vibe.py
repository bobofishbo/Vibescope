import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from PIL import Image
import torch
import faiss
import seaborn as sns
import matplotlib.pyplot as plt
import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

text_model = SentenceTransformer('all-MiniLM-L6-v2')
image_model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
image_model.eval()

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@timer_decorator
def load_profiles(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

@timer_decorator
def extract_text_features(profile):
    text_input = f"Name: {profile['name']}. Personality: {profile['personality']}"
    return text_model.encode(text_input)

@timer_decorator
def extract_image_features_batch(image_paths):
    images = [image_transforms(Image.open(path).convert("RGB")) for path in image_paths]
    image_tensor = torch.stack(images)  
    with torch.no_grad():
        features = image_model(image_tensor)
    return features.numpy()

@timer_decorator
def reduce_dimensionality(features):
    n_samples, n_features = features.shape
    n_components = min(n_samples, n_features, 50) 
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

@timer_decorator
def process_profiles(profiles):
    text_features = []
    image_paths = []
    for profile in profiles:
        text_features.append(extract_text_features(profile))
        image_paths.append(profile["image_path"])
    text_features = reduce_dimensionality(np.array(text_features))
    image_features = reduce_dimensionality(extract_image_features_batch(image_paths))
    return text_features, image_features

@timer_decorator
def calculate_similarity_matrix(text_features, image_features, text_weight=0.7, image_weight=0.3):
    text_similarity_matrix = cosine_similarity(text_features)
    image_similarity_matrix = cosine_similarity(image_features)
    
    if len(text_similarity_matrix) <= 20:
        visualize_matrix(text_similarity_matrix, "Text Similarity")
        visualize_matrix(image_similarity_matrix, "Image Similarity")
    
    return (
        text_weight * text_similarity_matrix +
        image_weight * image_similarity_matrix
    )

def visualize_matrix(matrix, title):
    df = pd.DataFrame(matrix)
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"{title} Matrix")
    plt.show()

@timer_decorator
def find_best_matches_faiss(features, profiles, k=5, similarity_threshold=0.1):
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    
    k = min(len(profiles), k)
    distances, indices = index.search(features, k)
    
    print("FAISS Distances (raw):", distances)  
    
    matches = []
    for i, profile in enumerate(profiles):
        similar_profiles = []
        for rank, j in enumerate(indices[i]):
            if i == j or distances[i][rank] == np.finfo(np.float32).max:  
                continue
            distance = distances[i][rank]
            normalized_distance = distance / (np.max(distances) if np.max(distances) > 0 else 1)
            similarity = 1 - normalized_distance  
            
            if similarity > similarity_threshold: 
                print(f"Profile: {profile['name']}, Match: {profiles[j]['name']}, Distance: {distance}, Normalized Distance: {normalized_distance}, Similarity: {similarity}")
                similar_profiles.append((profiles[j]["name"], similarity))
        
        similar_profiles.sort(key=lambda x: x[1], reverse=True)
        matches.append((profile["name"], similar_profiles))
    
    return matches

def main():
    profiles = load_profiles('profiles.json')
    text_features, image_features = process_profiles(profiles)
    combined_features = np.hstack((text_features, image_features))
    matched_groups = find_best_matches_faiss(combined_features, profiles)
    
    for profile_name, matches in matched_groups:
        print(f"{profile_name}'s best matches:")
        for match_name, score in matches:
            print(f"  - {match_name} (Score: {score:.2f})")

if __name__ == "__main__":
    main()
