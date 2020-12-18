from tqdm import tqdm

def train(net, trainloader, optimizer , criterion , device, NUM_EPOCHS):
    net = net.to(device)
    criterion = criterion.to(device)
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        with tqdm(trainloader,unit='batch') as tepoch:
            running_loss = 0.0
            for  (image,image_3d ) in tepoch:

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
        state = {
            'epoch': epoch + 1,
            'valid_loss_min': loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        } 
        print("\nSaving model...\n")
        save_model(state, False, best_model_dir)
        
        
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss))
        
        min_loss = min(train_loss) 
        min_loss_epoch = train_loss.index(min(train_loss)) + 1
        print("Best model found at epoch {} with a loss of: {}".format(min_loss_epoch, min_loss))

    return train_loss
