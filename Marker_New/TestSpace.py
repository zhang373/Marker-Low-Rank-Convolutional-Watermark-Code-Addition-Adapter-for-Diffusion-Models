import deeplake
import torchvision
import torch
ds = deeplake.load("hub://activeloop/tiny-imagenet-train")
dataloader = ds.pytorch(num_workers=4, batch_size=4, shuffle=False)
print(ds.tensors.keys())  # dict_keys(['images', 'labels'])
print(ds.images[0].shape[2])
print("Type:  ",type(ds.images[0]))
print(torchvision.transforms.functional.get_image_num_channels(torch.from_numpy(ds.images[0].numpy())))
print(torchvision.transforms.functional.get_image_num_channels(torch.tensor(ds.images[0])))