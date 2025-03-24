import torch
import open_clip
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define Dataset
class ImageTextDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        text = self.data.iloc[idx]['text']
        binary_label = self.data.iloc[idx]['binary_label']
        indoor_label = self.data.iloc[idx]['indoor_label']
        outdoor_label = self.data.iloc[idx]['outdoor_label']
        return image, text, binary_label, indoor_label, outdoor_label

# Define MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.dropout(torch.relu(x))
        x = self.fc2(x)
        return x

def main(hidden_dim):
    # Load dataset
    train_df = pd.read_csv("/home/shawnluo/open_clip/Training_Set.csv")
    val_df = pd.read_csv("/home/shawnluo/open_clip/Validation_Set.csv")
    test_df = pd.read_csv("/home/shawnluo/open_clip/Test_Set.csv")

    # Mapping labels
    train_df['text'] = train_df['text'].fillna("").astype(str)
    val_df['text'] = val_df['text'].fillna("").astype(str)
    test_df['text'] = test_df['text'].fillna("").astype(str)

    binary_label_mapping = {"indoor": 0, "outdoor": 1}
    indoor_label_mapping = {
        "indoor_crafts": 0, "indoor_blocks": 1, "indoor_news": 2,
        "indoor_other": 3, "indoor_puzzle": 4, "indoor_snack": 5, "indoor_story": 6
    }
    outdoor_label_mapping = {
        "outdoor_grass": 0, "outdoor_hood": 1, "outdoor_patio": 2,
        "outdoor_other": 3, "outdoor_sand": 4
    }

    for df in [train_df, val_df, test_df]:
        df['binary_label'] = df['label'].apply(lambda x: binary_label_mapping['indoor'] if 'indoor' in x else binary_label_mapping['outdoor'])
        df['indoor_label'] = df['label'].map(indoor_label_mapping)
        df['outdoor_label'] = df['label'].map(outdoor_label_mapping)

    # Load open_clip
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = False if "visual.transformer.resblocks.11" not in name else True

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        preprocess_train
    ])

    train_dataset = ImageTextDataset(train_df, transform=transform)
    val_dataset = ImageTextDataset(val_df, transform=preprocess_val)
    test_dataset = ImageTextDataset(test_df, transform=preprocess_val)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    image_feature_dim = model.visual.output_dim
    text_feature_dim = model.token_embedding.embedding_dim

    binary_classifier = MLPClassifier(input_dim=image_feature_dim + text_feature_dim, hidden_dim=hidden_dim, num_classes=2)
    indoor_classifier = MLPClassifier(input_dim=image_feature_dim + text_feature_dim, hidden_dim=hidden_dim, num_classes=len(indoor_label_mapping))
    outdoor_classifier = MLPClassifier(input_dim=image_feature_dim + text_feature_dim, hidden_dim=hidden_dim, num_classes=len(outdoor_label_mapping))
    binary_classifier.to(device)
    indoor_classifier.to(device)
    outdoor_classifier.to(device)

    optimizer_binary = optim.AdamW(binary_classifier.parameters(), lr=1e-3)
    optimizer_indoor = optim.AdamW(indoor_classifier.parameters(), lr=1e-3)
    optimizer_outdoor = optim.AdamW(outdoor_classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    def train_multistage_classifiers(num_epochs=50):
        nonlocal best_val_loss

        for epoch in range(num_epochs):
            binary_classifier.train()
            indoor_classifier.train()
            outdoor_classifier.train()

            total_binary_loss, total_binary_correct, total_binary_samples = 0.0, 0, 0
            total_indoor_loss, total_indoor_correct, total_indoor_samples = 0.0, 0, 0
            total_outdoor_loss, total_outdoor_correct, total_outdoor_samples = 0.0, 0, 0

            for images, texts, binary_labels, indoor_labels, outdoor_labels in tqdm(train_dataloader):
                images, texts = images.to(device), tokenizer(texts).to(device)
                binary_labels = binary_labels.to(device)
                indoor_labels, outdoor_labels = indoor_labels.to(device), outdoor_labels.to(device)

                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                features = torch.cat((image_features, text_features), dim=1)

                # Binary Classification
                binary_outputs = binary_classifier(features)
                binary_loss = criterion(binary_outputs, binary_labels)

                optimizer_binary.zero_grad()
                binary_loss.backward(retain_graph=True)
                optimizer_binary.step()

                total_binary_loss += binary_loss.item()
                binary_preds = binary_outputs.argmax(dim=1)
                total_binary_correct += (binary_preds == binary_labels).sum().item()
                total_binary_samples += binary_labels.size(0)

                # Indoor Classification
                indoor_mask = (binary_labels == 0)
                if indoor_mask.sum() > 0:
                    indoor_features = features[indoor_mask]
                    indoor_labels_filtered = indoor_labels[indoor_mask].long()
                    indoor_outputs = indoor_classifier(indoor_features)
                    indoor_loss = criterion(indoor_outputs, indoor_labels_filtered)

                    optimizer_indoor.zero_grad()
                    indoor_loss.backward(retain_graph=True)
                    optimizer_indoor.step()

                    total_indoor_loss += indoor_loss.item()
                    indoor_preds = indoor_outputs.argmax(dim=1)
                    total_indoor_correct += (indoor_preds == indoor_labels_filtered).sum().item()
                    total_indoor_samples += indoor_labels_filtered.size(0)

                # Outdoor Classification
                outdoor_mask = (binary_labels == 1)
                if outdoor_mask.sum() > 0:
                    outdoor_features = features[outdoor_mask]
                    outdoor_labels_filtered = outdoor_labels[outdoor_mask].long()
                    outdoor_outputs = outdoor_classifier(outdoor_features)
                    outdoor_loss = criterion(outdoor_outputs, outdoor_labels_filtered)

                    optimizer_outdoor.zero_grad()
                    outdoor_loss.backward()
                    optimizer_outdoor.step()

                    total_outdoor_loss += outdoor_loss.item()
                    outdoor_preds = outdoor_outputs.argmax(dim=1)
                    total_outdoor_correct += (outdoor_preds == outdoor_labels_filtered).sum().item()
                    total_outdoor_samples += outdoor_labels_filtered.size(0)

            # Calculate overall metrics for the epoch
            avg_binary_loss = total_binary_loss / len(train_dataloader)
            avg_binary_acc = total_binary_correct / total_binary_samples
            avg_indoor_loss = total_indoor_loss / len(train_dataloader)
            avg_indoor_acc = total_indoor_correct / total_indoor_samples if total_indoor_samples > 0 else 0
            avg_outdoor_loss = total_outdoor_loss / len(train_dataloader)
            avg_outdoor_acc = total_outdoor_correct / total_outdoor_samples if total_outdoor_samples > 0 else 0

            print(f"Epoch [{epoch + 1}/{num_epochs}], Binary Loss: {avg_binary_loss:.4f}, "
                f"Binary Acc: {avg_binary_acc:.4f}, Indoor Loss: {avg_indoor_loss:.4f}, "
                f"Indoor Acc: {avg_indoor_acc:.4f}, Outdoor Loss: {avg_outdoor_loss:.4f}, "
                f"Outdoor Acc: {avg_outdoor_acc:.4f}")

            # Validate and save the best models
            val_loss = validate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(binary_classifier.state_dict(), '/home/shawnluo/open_clip/Best_binary_model_path')
                torch.save(indoor_classifier.state_dict(), '/home/shawnluo/open_clip/Best_indoor_model_path')
                torch.save(outdoor_classifier.state_dict(), '/home/shawnluo/open_clip/Best_outdoor_model_path')
                print(f"Best models saved at epoch {epoch + 1}")


    def validate():
        binary_classifier.eval()
        indoor_classifier.eval()
        outdoor_classifier.eval()

        total_val_binary_loss, total_val_binary_correct, total_val_samples = 0.0, 0, 0
        total_val_indoor_loss, total_val_indoor_correct, total_val_indoor_samples = 0.0, 0, 0
        total_val_outdoor_loss, total_val_outdoor_correct, total_val_outdoor_samples = 0.0, 0, 0

        with torch.no_grad():
            for images, texts, binary_labels, indoor_labels, outdoor_labels in tqdm(val_dataloader):
                images, texts = images.to(device), tokenizer(texts).to(device)
                binary_labels = binary_labels.to(device)
                indoor_labels, outdoor_labels = indoor_labels.to(device), outdoor_labels.to(device)

                # Extract features
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                features = torch.cat((image_features, text_features), dim=1)

                # Binary Classification
                binary_outputs = binary_classifier(features)
                binary_loss = criterion(binary_outputs, binary_labels)
                total_val_binary_loss += binary_loss.item()

                binary_preds = binary_outputs.argmax(dim=1)
                total_val_binary_correct += (binary_preds == binary_labels).sum().item()
                total_val_samples += binary_labels.size(0)

                # Indoor Classification
                indoor_mask = (binary_preds == 0)  # binary label "indoor"
                if indoor_mask.sum() > 0:
                    indoor_features = features[indoor_mask]
                    indoor_labels_filtered = indoor_labels[indoor_mask].long()

                    # Ensure no invalid labels exist
                    valid_indices = (indoor_labels_filtered >= 0) & (indoor_labels_filtered < len(indoor_label_mapping))
                    if valid_indices.sum() > 0:  # Proceed only if there are valid labels
                        indoor_features = indoor_features[valid_indices]
                        indoor_labels_filtered = indoor_labels_filtered[valid_indices]

                        indoor_outputs = indoor_classifier(indoor_features)
                        indoor_loss = criterion(indoor_outputs, indoor_labels_filtered)
                        total_val_indoor_loss += indoor_loss.item()

                        indoor_preds = indoor_outputs.argmax(dim=1)
                        total_val_indoor_correct += (indoor_preds == indoor_labels_filtered).sum().item()
                        total_val_indoor_samples += indoor_labels_filtered.size(0)

                # Outdoor Classification
                outdoor_mask = (binary_preds == 1)  # binary label "outdoor"
                if outdoor_mask.sum() > 0:
                    outdoor_features = features[outdoor_mask]
                    outdoor_labels_filtered = outdoor_labels[outdoor_mask].long()

                    # Ensure no invalid labels exist
                    valid_indices = (outdoor_labels_filtered >= 0) & (outdoor_labels_filtered < len(outdoor_label_mapping))
                    if valid_indices.sum() > 0:  # Proceed only if there are valid labels
                        outdoor_features = outdoor_features[valid_indices]
                        outdoor_labels_filtered = outdoor_labels_filtered[valid_indices]

                        outdoor_outputs = outdoor_classifier(outdoor_features)
                        outdoor_loss = criterion(outdoor_outputs, outdoor_labels_filtered)
                        total_val_outdoor_loss += outdoor_loss.item()

                        outdoor_preds = outdoor_outputs.argmax(dim=1)
                        total_val_outdoor_correct += (outdoor_preds == outdoor_labels_filtered).sum().item()
                        total_val_outdoor_samples += outdoor_labels_filtered.size(0)

        # Calculate overall metrics for the validation set
        avg_val_binary_loss = total_val_binary_loss / len(val_dataloader)
        avg_val_binary_acc = total_val_binary_correct / total_val_samples
        avg_val_indoor_loss = total_val_indoor_loss / len(val_dataloader)
        avg_val_indoor_acc = total_val_indoor_correct / total_val_indoor_samples if total_val_indoor_samples > 0 else 0
        avg_val_outdoor_loss = total_val_outdoor_loss / len(val_dataloader)
        avg_val_outdoor_acc = total_val_outdoor_correct / total_val_outdoor_samples if total_val_outdoor_samples > 0 else 0

        print(f"Validation - Binary Loss: {avg_val_binary_loss:.4f}, Binary Acc: {avg_val_binary_acc:.4f}, "
            f"Indoor Loss: {avg_val_indoor_loss:.4f}, Indoor Acc: {avg_val_indoor_acc:.4f}, "
            f"Outdoor Loss: {avg_val_outdoor_loss:.4f}, Outdoor Acc: {avg_val_outdoor_acc:.4f}")

        return avg_val_binary_loss



    def test_multistage_classifiers():
        binary_classifier.load_state_dict(torch.load('/home/shawnluo/open_clip/Best_binary_model_path'))
        indoor_classifier.load_state_dict(torch.load('/home/shawnluo/open_clip/Best_indoor_model_path'))
        outdoor_classifier.load_state_dict(torch.load('/home/shawnluo/open_clip/Best_outdoor_model_path'))

        binary_classifier.eval()
        indoor_classifier.eval()
        outdoor_classifier.eval()

        all_binary_preds, all_binary_labels = [], []
        all_indoor_preds, all_indoor_labels = [], []
        all_outdoor_preds, all_outdoor_labels = [], []

        with torch.no_grad():
            for images, texts, binary_labels, indoor_labels, outdoor_labels in tqdm(test_dataloader):
                images, texts = images.to(device), tokenizer(texts).to(device)
                binary_labels = binary_labels.to(device)
                indoor_labels, outdoor_labels = indoor_labels.to(device), outdoor_labels.to(device)

                # Extract features
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                features = torch.cat((image_features, text_features), dim=1)

                # Binary Classification
                binary_outputs = binary_classifier(features)
                binary_preds = binary_outputs.argmax(dim=1)
                all_binary_preds.extend(binary_preds.cpu().numpy())
                all_binary_labels.extend(binary_labels.cpu().numpy())

                # Indoor Classification
                indoor_mask = (binary_preds == 0)
                if indoor_mask.sum() > 0:
                    indoor_features = features[indoor_mask]
                    indoor_outputs = indoor_classifier(indoor_features)
                    indoor_preds = indoor_outputs.argmax(dim=1)
                    all_indoor_preds.extend(indoor_preds.cpu().numpy())
                    all_indoor_labels.extend(indoor_labels[indoor_mask].cpu().numpy())

                # Outdoor Classification
                outdoor_mask = (binary_preds == 1)
                if outdoor_mask.sum() > 0:
                    outdoor_features = features[outdoor_mask]
                    outdoor_outputs = outdoor_classifier(outdoor_features)
                    outdoor_preds = outdoor_outputs.argmax(dim=1)
                    all_outdoor_preds.extend(outdoor_preds.cpu().numpy())
                    all_outdoor_labels.extend(outdoor_labels[outdoor_mask].cpu().numpy())

        # Confusion matrices
        plot_confusion_matrices(all_binary_labels, all_binary_preds, binary_label_mapping, "Binary Classification")
        plot_confusion_matrices(all_indoor_labels, all_indoor_preds, indoor_label_mapping, "Indoor Classification")
        plot_confusion_matrices(all_outdoor_labels, all_outdoor_preds, outdoor_label_mapping, "Outdoor Classification")

    def plot_confusion_matrices(true_labels, predictions, label_mapping, title):
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {title}')
        plt.savefig(f'{title.replace(" ", "_").lower()}_confusion_matrix.png')
        plt.show()

    # Train and test
    train_multistage_classifiers(num_epochs=50)
    test_multistage_classifiers()

if __name__ == "__main__":
    print("Training MLP with hidden dimension 32")
    main(hidden_dim=32)
