from model import *
from prepare_dataset import *
import torchvision
from torch.utils.data import DataLoader
from test import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

batch_size = 64

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform).data
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DiffusionModel(T=4000, embed_size=64, n_channels=1, n_classes=1).to(device)

criterion = nn.HuberLoss()
optimizer = torch.optim.NAdam(model.parameters(), lr=2e-4)

epochs = 50
for epoch in range(epochs):
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        dict_, noise = prepare_batch(batch)
        X_noisy = dict_["X_noisy"].to(device)
        time = dict_["time"].to(device)

        output = model(X_noisy, time).squeeze(1).unsqueeze(-1)
        noise = noise.to(device)
        
        optimizer.zero_grad()
        loss = criterion(output, noise)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200}')
            running_loss = 0.0
            
PATH = './Diffusion_Model.pth'
torch.save(model.state_dict(), PATH)

X_gen = generate(model)
plot_multiple_images(X_gen.to(device='cpu').numpy(), 8)
plt.show()
