import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util

def read_classes(file_path):

    fp = open(file_path, "r")
    classes = fp.readline()
    classes = classes. split(",")
    fp.close()

    return classes

classes = read_classes('classes.txt')

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.dir_A = os.path.join(opt.dataroot,'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot,'B')  # create a path '/path/to/data/trainB'
        top_label = ['A', 'B']

        sec_label = os.listdir(self.dir_A)
        total_data = []
        data = []
        for t in top_label: 
            data = []
            for s in sec_label:
                sec_folder_dir = os.path.join(opt.dataroot, t, s)
                img_list = os.listdir(sec_folder_dir)
                for img in img_list:
                    final_path = os.path.join(sec_folder_dir, img)
                    # open image
                    data.append((final_path, s))
            total_data.append(data)

        # if opt.phase == "test" and not os.path.exists(self.dir_A) \
        #    and os.path.exists(os.path.join(opt.dataroot, "valA")):
        #     self.dir_A = os.path.join(opt.dataroot, "valA")
        #     self.dir_B = os.path.join(opt.dataroot, "valB")
        # self.A_data_pair = total_data[0]
        # self.B_data_pair = total_data[1]

        #self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        #self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
       
        self.A_paths = [p for p, l in total_data[0]]
        self.B_paths = [p for p, l in total_data[1]]
        self.A_label = [l for p, l in total_data[0]]
        self.B_label = [l for p, l in total_data[1]]

        #print(self.A_label,self.B_label)
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)
        
        A_label = classes.index(self.A_label[index])
        B_label = classes.index(self.B_label[index_B])
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_label': A_label, 'B_label': B_label}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
