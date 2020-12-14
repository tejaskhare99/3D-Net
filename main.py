from training_func import train
from DataLoader import get_device, make_dir
from network import net
from testing_func import test_image_reconstruction
import torch 


device = get_device()
print(device)
# load the neural network onto the device
net.to(device)
make_dir()
# train the network
train_loss = train(model, trainloader, NUM_EPOCHS)
torch.save(model.state_dict(), '/content')
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('deep_ae_mnist_loss.png')
# test the network
test_image_reconstruction(model, testloader)
