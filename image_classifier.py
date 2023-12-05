import torch # PyTorch package
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import json

# Load the configuration file
try:
    with open('config.json') as config_file: 
        config = json.load(config_file)
except FileNotFoundError: 
    print("Error: The configuration file 'config.json' was not found.")
    exit(1)
except json.JSONDecodeError: 
    print("Error: The configuration file 'config.json' is not a valid JSON.")
    exit(1) 
  
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_progress(iteration, total, prefix='Progress:', suffix='Complete', length=50): 
    percent = "{:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total) 
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
    if iteration == total: 
        print()
        
def get_mean_and_std(loader): 
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for i, (images, _) in enumerate(loader):
        # Print progress
        print_progress(i + 1, len(loader))

        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std

def create_transforms(mean, std, augment=False): 
    if augment: 
        return transforms.Compose([
            transforms.RandomResizedCrop(config['image_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else: 
        return transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
def create_data_loader(data_dir, transform, shuffle=True): 
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle, num_workers=config['num_workers'])
    return loader

def load_dataset(): 
    # Loading the dataset without normalization to calculate the mean and std
    preprocess = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])), 
        transforms.ToTensor()
    ])
    pre_dataset = datasets.ImageFolder(root=config['data']['train_dir'], transform=preprocess)
    pre_loader = DataLoader(pre_dataset, batch_size=config['batch_size'], shuffle=True)
    
    mean, std = get_mean_and_std(pre_loader)
    
    # Data agumentation for training
    train_transforms = create_transforms(mean, std, augment=True)
    train_loader = create_data_loader(config['data']['train_dir'], train_transforms)
    
    # Data augmentation for validation
    valid_transforms = create_transforms(mean, std, augment=False)
    valid_loader = create_data_loader(config['data']['valid_dir'], valid_transforms) 
    
    return train_loader, valid_loader

# The CNN
# Define a convolution neural network
class Network(nn.Module):
    def __init__(self, num_classes, conv_config, image_size):
        super(Network, self).__init__()
        
       # Convolutional layers
        self.conv1 = nn.Conv2d(conv_config['in_channels_conv1'], conv_config['out_channels_conv1'], conv_config['kernal_size'], conv_config['stride'], conv_config['padding'])
        self.bn1 = nn.BatchNorm2d(conv_config['out_channels_conv1'])
        self.conv2 = nn.Conv2d(conv_config['in_channels_conv2'], conv_config['out_channels_conv2'], conv_config['kernal_size'], conv_config['stride'], conv_config['padding'])
        self.bn2 = nn.BatchNorm2d(conv_config['out_channels_conv2'])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(conv_config['in_channels_conv3'], conv_config['out_channels_conv3'], conv_config['kernal_size'], conv_config['stride'], conv_config['padding'])
        self.bn3 = nn.BatchNorm2d(conv_config['out_channels_conv3'])  # Note: Adjusted to in_channels_conv3
        self.conv4 = nn.Conv2d(conv_config['in_channels_conv4'], conv_config['out_channels_conv4'], conv_config['kernal_size'], conv_config['stride'], conv_config['padding'])
        self.bn4 = nn.BatchNorm2d(conv_config['out_channels_conv4'])  # Note: Adjusted to in_channels_conv4
        
         # Fully connected layer
        self.fc1 = nn.Linear(self._calc_fc_input(image_size), num_classes)
        

    def _calc_fc_input(self, image_size):
        # Temporarily create an input tensor to calculate the size of the output after conv layers
        sample = torch.zeros(1, 3, image_size, image_size)
        sample = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(sample)))))))
        sample = F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(sample))))))
        return int(np.prod(sample.size()))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Function to save the model
def saveModel(model, path):
    torch.save(model.state_dict(), path)
    
# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(model, test_loader):
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
       
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images.to(DEVICE))
            outputs = outputs.cpu()
            
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(model, train_loader, number_of_epochs, loss_fn, optimizer, test_loader, verbose):
    best_accuracy = 0.0

    for epoch in range(number_of_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if verbose and (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{number_of_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 1000:.4f}')
                running_loss = 0.0

        model.eval()
        accuracy = testAccuracy(model, test_loader)  # Ensure this function calculates accuracy on a separate validation set
        
        if verbose:
            print(f'For epoch {epoch + 1} the test accuracy over the whole test set is {accuracy} %')

        if accuracy > best_accuracy:
            saveModel(model, "./myFirstModel.pth")  # Consider also saving the optimizer state and epoch number
            best_accuracy = accuracy

# Function to test the model with a batch of images and show the labels predictions
def testBatch(model, test_loader, classes, batch_size):
    model.eval() # Sets the model to evaluation mode
    
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # Move the images and labels to the same device as the model
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    
    # Show the real labels on the screen 
    real_labels_indice = labels.tolist()
    
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
        for j in range(batch_size)))
    
     # Move the images to the device and get the outputs from the model
    with torch.no_grad(): 
        outputs = model(images)
        _, predicted_indices = torch.max(outputs, 1)
        predicted_indices = predicted_indices.tolist()

        # Convert the predicted indices to class names
        predicted_labels = [classes[idx] for idx in predicted_indices]
        print('Predicted: ', ' '.join(predicted_labels))
        
        # Calculate the batch accuracy
        correct = sum(p == t for p, t in zip(predicted_indices, real_labels_indice))
        accuracy = correct / batch_size * 100
        print(f'Accuracy of the batch: {accuracy:.2f} %')
        
# Function to test what classes performed well
def calculate_class_accuracy(model, test_loader, num_of_classes):
    model.eval()
    class_correct = torch.zeros(num_of_classes, device=DEVICE)
    class_total = torch.zeros(num_of_classes, device=DEVICE)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = torch.eq(predicted, labels)

            for i in range(num_of_classes):
                class_mask = labels == i
                class_correct[i] += correct[class_mask].sum()
                class_total[i] += class_mask.sum()

    return class_correct / class_total

def print_class_accuracy(class_accuracy, classes):
    for i, accuracy in enumerate(class_accuracy):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * accuracy))
   
def compute_confusion_matrix(model, test_loader, classes): 
    all_labels = []
    all_preds = []
    
    model.eval()
    
    with torch.no_grad(): 
        for images, labels in test_loader: 
            images = images.to(next(model.parameters()).device)
            labels = labels.to(DEVICE) 
            outputs = model(images) 
            _, preds = torch.max(outputs, 1)
            
            # Move to CPU for numpy compatibility
            preds = preds.cpu().numpy() 
            labels = labels.cpu().numpy()
            
            # Extend the list
            all_labels.extend(labels)
            all_preds.extend(preds)
            
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds) 
    
    # Calculate overall precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    plot_confusion_matrix(conf_matrix, classes)
    print_evaluation_scores(precision, recall, f1)
            
def plot_confusion_matrix(conf_matrix, classes): 
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()  

def print_evaluation_scores(precision, recall, f1): 
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    
    try: 
        train_loader, test_loader = load_dataset()
    except Exception as e: 
        print(f"Error loading dataset: {e}")
        exit(1) 

    # Instantiate a neural network model
    model = Network(num_classes=config['num_classes'], 
                conv_config=config['conv_config'], 
                image_size=config['image_size']).to(DEVICE)

    # Loss function with Classification corss-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    
    try: 
        train(model, train_loader, config['epochs'], loss_fn, optimizer, test_loader, verbose=True) # Detailed output
        print('Finished Training')
    except RuntimeError as e: 
        print(f"Error during training: {e}")
        exit(1)
    
    # Save the model
    try: 
       model.load_state_dict(torch.load(config['model_path']))
    except FileNotFoundError: 
        print("Error: The model file was not found")
        exit(1)
    except RuntimeError:
        print("Error: The model file is not compatible or is corrupted.")
        exit(1)
    
    # Evaluate the mode
    testAccuracy(model, test_loader)
    testBatch(model, test_loader, config['classes'], config['batch_size'] )
    
    class_accuracy = calculate_class_accuracy(model, test_loader, config['num_classes'])
    print_class_accuracy(class_accuracy, config['classes'])
    
    compute_confusion_matrix(model, test_loader, config['classes'])
    
    # Optinally load the model if it is a new execution environment
    # model.load_state_dict(torch.load(config['model_path']))
