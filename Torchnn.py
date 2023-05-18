# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
            #1,28,28 - classes 0-9
            # Pour mercredi 



# Image Classifier Neural Network
class SignClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )


        self.conv1 =  nn.Conv2d(1, 32, (3,3))
        self.RelU =   nn.ReLU()
        self.conv2 =  nn.Conv2d(32, 64, (3,3)) 
        self.conv3 =  nn.Conv2d(64, 64, (3,3))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64*(28-6)*(28-6), 10)

        #
        # 800 *800

        # partager en 8 models


    def forward(self, x):
        out = self.conv1(x)
        out = self.RelU(out)
        out = self.conv2(out)
        out =self.RelU(out)
        out= self.conv3(out)
        out = self.RelU(out)
        out= self.flatten(out)
        print(out.shape)
        out = self.linear(out)
        return out
        #return self.model(x)


# Instance of the neural network, loss, optimizer 
clf = SignClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 


# Training flow 
# if __name__ == "__main__": 
#     for epoch in range(10): # train for 10 epochs
#         for batch in dataset: 
#             X,y = batch 
#             X, y = X.to('cpu'), y.to('cpu')   #send to GPU
#             yhat = clf(X)                       #predection
#             loss = loss_fn(yhat, y) 

#             # Apply backprop 
#             opt.zero_grad()
#             loss.backward() 
#             opt.step() 

#         print(f"Epoch:{epoch} loss is {loss.item()}")
       
       
       
#     # Create & Save Our model .pt   
#     with open('model_state.pt', 'wb') as f: 
#         save(clf.state_dict(), f) 



#First Test 
#if __name__ == "__main__":
  # with open('model_state.pt', 'rb') as f:
#    clf.load_state_dict(load(f))

img = Image.open('img_1.jpg')
img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
   
print(torch.argmax(clf(img_tensor)))