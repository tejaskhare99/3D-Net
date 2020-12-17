import torch


from tqdm import tqdm


def test(net, trainloader, criterion, device):
   net = net.to(device)
   criterion = criterion.to(device)
   train_loss = []
   with torch.no_grad:
      with tqdm(trainloader, unit='batch') as tepoch:
         running_loss = 0.0
         for (image, image_3d) in tepoch:
            img = image.to(device).float()

            image_3d = image_3d.to(device)

            # print(img.shape)


            outputs = net(img)
            loss = criterion(outputs, image_3d)
            loss.backward()

            running_loss += loss.item()

      loss = running_loss / len(trainloader)
      train_loss.append(loss)
      print(' Test Loss: {:.3f}'.format(
          loss))

   return train_loss