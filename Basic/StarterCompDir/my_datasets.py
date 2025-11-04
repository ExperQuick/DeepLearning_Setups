from plf.utils import Component

from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Component, Dataset):
    def __init__(self):
        Dataset.__init__(self)
        Component.__init__(self)
        self.args = {'root_dir', 'train', 'transform'}

    def _setup(self,args):
        # Load MNIST (PyTorch will handle raw files)
        self.mnist = datasets.MNIST(root=args['root_dir'], train=args['train'], download=True)
        self.transform = self.load_component(**args['transform'])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class Augment(Component):
    def __init__(self):
        super().__init__()
        self.args = {}

    def _setup(self,args):
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
    def __call__(self, img):
        return self.transform(img)

