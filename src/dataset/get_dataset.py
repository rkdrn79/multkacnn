from torchvision import datasets, transforms


from src.dataset.datacollator import VisionDataCollator

def get_dataset(args):
    if args.data_name == "mnist":

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)

        data_collator = VisionDataCollator(args)

    if args.data_name == 'fashion_mnist':

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=transform)
        val_dataset = datasets.FashionMNIST(args.data_path, train=False, download=True, transform=transform)

        data_collator = VisionDataCollator(args)

    return train_dataset, val_dataset, data_collator