import torch
import torchvision
import torch.utils.data
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image
from VAE import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", "GPU" if torch.cuda.is_available() else "CPU", " for model training")
### MNIST Load Code from pytorch documentation
from torchvision import transforms, datasets

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
				   transform=transforms.ToTensor()),
	batch_size=100, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
	batch_size=100, shuffle=True, **kwargs)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_function = VAE.vae_loss

def train(epoch):
	model.train()
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		data = data.to(device)
		optimizer.zero_grad()
		recon_batch, mu, logvar = model(data)
		loss = loss_function(recon_batch, data, mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item() / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for i, (data, _) in enumerate(test_loader):
			data = data.to(device)
			recon_batch, mu, logvar = model(data)
			test_loss += loss_function(recon_batch, data, mu, logvar).item()
			if i == 0:
				n = min(data.size(0), 8)
				comparison = torch.cat([data[:n],
									  recon_batch.view(100, 1, 28, 28)[:n]])
				save_image(comparison.cpu(),
						 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))

sample = torch.randn(64, 10).to(device)

for epoch in range(1, 50 + 1):
	train(epoch)
	test(epoch)
	with torch.no_grad():
		results = model.decode(sample).cpu()
		save_image(results.view(64, 1, 28, 28),
				   'results/sample_' + str(epoch) + '.png')

torch.save(model.state_dict(), './vae_model.pt')
torch.save(optimizer.state_dict(), './vae_optim.pt')


print("Model and optimizer are loaded")
