import torch
import torch.optim as Optim


def loss_func(point):
    return torch.mean(torch.abs(point - torch.tensor([0.0, 0.0], requires_grad=True)))


if __name__ == '__main__':
    point = torch.tensor([2.0, 2.0], requires_grad=True)

    optimizer = Optim.SGD([point], lr=0.1)
    lr_schedule = Optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    for epoch in range(500):
        optimizer.zero_grad()
        loss = loss_func(point)
        loss.backward()
        lr_schedule.step()
        optimizer.step()
        print("epoch: {}, point: ({},{})".format(epoch, point[0].item(), point[1].item()))


