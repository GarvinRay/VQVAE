from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(dataset_type, batch_size, img_shape=None, num_workers=2):
    if dataset_type.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(img_shape or (28, 28)),
            transforms.ToTensor()
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_type.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(img_shape or (32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset_type')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)