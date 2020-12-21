from tqdm import tqdm
from utils import plot3d

def train(net, trainloader, optimizer, criterion, device, NUM_EPOCHS):
    net = net.to(device)
    criterion = criterion.to(device)
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        with tqdm(trainloader, unit='batch') as tepoch:
            running_loss = 0.0
            for (image, image_3d) in tepoch:

                img = image.to(device).float()

                image_3d = image_3d.to(device)

                # print(img.shape)

                optimizer.zero_grad()
                outputs = net(img)
                loss = criterion(outputs, image_3d)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                plot3d(outputs[0].detatch().numpy())


        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss))

    return train_loss
