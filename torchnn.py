import torch
from PIL import Image
from torch import nn,save,load              #Contains Neural Network classes
from torch.optim import Adam                #Adam is a adaptive moment estimation optimizer
from torch.utils.data import DataLoader     #To load a dataset from pytorch
from torchvision import datasets
from torchvision.transforms import ToTensor #To transform images into tensors


## GET DATA

# MNIST dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9

train = datasets.MNIST(root="data",download=True,transform=ToTensor())
#dimension:(1,28,28)
#contains the data for training the model
#root - specifies the location where the data will be downloaded
#download - download status
#transform - specifies to which object to transform

dataset = DataLoader(train,32)      #Parse out the trained partition
                                    #Converting into batches of 32 images

#Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(

            #1st Layer
            nn.Conv2d(1,32,(3,3)),      #Only 1 channel for B/W images & 32 Filters of 3x3 px Kernels
            nn.ReLU(),                  #Activation to handle non-linearity

            #2nd Layer
            nn.Conv2d(32,64,(3,3)),     #32 i/p channels here
            nn.ReLU(),

            #3rd Layer
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),

            #In each layer we are shaving off 2 px
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10),     #10 o/p
        )

    def forward(self,x):
        return self.model(x)

# Instance of the neural network, loss, optimizer

clf = ImageClassifier().to("cpu")
#CUDA == GPU, it allows general-purpose computing on GPU
#If your device don't have NVIDIA GPU then use "cpu" instead of "cuda"


opt = Adam(clf.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()

#Training Flow
if __name__ == "__main__":

    #Training

    for epoch in range(10):     #Train for 10 epoch
        for batch in dataset:
            X,y= batch
            X,y= X.to("cpu"), y.to("cpu")
            yhat = clf(X)
            loss = loss_fn(yhat,y)

            #Apply backprop
            opt.zero_grad()     #zero out any existing gradient
            loss.backward()
            opt.step()

        print(f"Epoch : {epoch} loss is {loss.item()}")

    with open("model_state.pt","wb") as file :
        save(clf.state_dict(),file)


    #Predicting

    with open("model_state.pt","rb") as file:
        clf.load_state_dict(load(file))

    for i in range(1,4,1):
        img = Image.open(f"img_{i}.jpg")
        img_tensor = ToTensor()(img).unsqueeze(0).to("cpu")
        print(torch.argmax(clf(img_tensor)))