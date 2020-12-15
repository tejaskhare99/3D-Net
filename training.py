def train(net, trainloader, optimizer , criterion , device, NUM_EPOCHS):
    net = net.to(device)
    criterion = criterion.to(device)
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i , (image,image_3d ) in enumerate(trainloader):

            img = image.to(device).float()

            image_3d = image_3d.to(device)

            # print(img.shape)
            
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, image_3d)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss))

    return train_loss
