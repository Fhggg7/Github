from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.conv import Net
from models.rnn_conv import ImageRNN
from models.alexnet import AlexNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import datetime

# functions to show an image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        # compute output
        print("train...")
        # data, target = data.to(device), target.to(device)
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        # loss = criterion(outputs, target)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))


def train_cnn(log_interval, model, device, train_loader, optimizer, epoch):
    if torch.cuda.is_available():
        print("cuda available")
    else:
        print("sike")
    print("train cnn...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print("test...")
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_rnn(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # reset hidden states
        model.hidden = model.init_hidden()
        data = data.view(-1, 28, 28)
        outputs = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        loss.backward(); optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, RNN):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if RNN:
                data = torch.squeeze(data)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    begin_time = datetime.datetime.now()
    epoches = 3
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = True

    #RNN
    RNN = True
    N_STEPS = 28
    N_INPUTS = 28
    N_NEURONS = 150
    N_OUTPUTS = 10

    if RNN:
        epoches = 15

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")


    ######################3   Torchvision    ###########################3
    # Use data predefined loader
    # Pre-processing by using the transform.Compose
    # divide into batches
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=1000, shuffle=True, **kwargs)

    # # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # # img = torchvision.utils.make_grid(images)
    # # imsave(img)
    #
    # # #####################    Build your network and run   ############################
    if RNN:
        model = AlexNet().to(device)
    else:
        model = Net().to(device)

    if RNN:
        optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epoches + 1):
        if RNN:
            train(log_interval, model, device, train_loader, optimizer, epoch)
        else:
            # train_cnn(train_loader,)
            train(log_interval, model, device, train_loader, optimizer, epoch)

        test(model, device, test_loader, RNN)
        scheduler.step()

    if save_model:
        if RNN:
            torch.save(model.state_dict(), "./results/mnist_rnn.pt")
        else:
            torch.save(model.state_dict(), "./results/mnist.pt")
    print(datetime.datetime.now() - begin_time)
    # testtrainedmodel(test_loader, device)


def testtrainedmodel(test_loader, device):
    testmodel = Net().to(device)
    testmodel.load_state_dict(torch.load("./results/mnist.pt"))
    testmodel.eval()

    for data, target in test_loader:
        plt.figure(figsize=(20, 10))
        output = testmodel(data.to(device))
        preds = torch.max(output.data, 1)
        # print(preds)
        images_so_far = 0
        # print(data.to(device).size()[0])
        # print("list " + str(len(preds)))

        for j in range(len(preds[0])):

            images_so_far += 1
            # plt.figure()
            plt.axis('off')
            plt.subplot(2, 10, images_so_far)
            # print(type(format(preds[1][j])))
            plt.title('predicted : {}'.format(preds[1][j]))
            # plt.tight_layout(pad=10)

            imshow(data[j])
            if images_so_far == 20:
                plt.axis('off')
                plt.show()
                # input("continue? ")
                images_so_far = 0


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.5)  # pause a bit so that plots are updated


if __name__ == '__main__':
    main()