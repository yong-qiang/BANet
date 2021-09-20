from PIL import Image
import torch
import numpy as np

cityscapes_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                      220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0,
                      70,
                      0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]


#camvid_palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 60, 40, 222, 128, 128, 0, 192, 128, 128, 64,
#                  64,
#                  128, 64, 0, 128, 64, 64, 0, 0, 128, 192]

camvid_palette = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0,
                  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]



mvd_palette = [165, 42, 42, 0, 192, 0, 196, 196, 196, 190, 153, 153, 180, 165, 180, 90, 120, 150, 102, 102, 156, 128, 64, 255, 140, 140, 200, 170, 170, 170, 250, 170, 160, 96, 96, 96,                230, 150, 140, 128, 64, 128, 110, 110, 110, 244, 35, 232, 150, 100, 100, 70, 70, 70, 150, 120, 90, 220, 20, 60, 255, 0, 0, 255, 0, 100, 255, 0, 200, 200, 128, 128,                     255, 255, 255, 64, 170, 64, 230, 160, 50, 70, 130, 180, 190, 255, 255, 152, 251, 152, 107, 142, 35, 0, 170, 30, 255, 255, 128, 250, 0, 30, 100, 140, 180, 220, 220, 220,                220, 128, 128, 222, 40, 40, 100, 170, 30, 40, 40, 40, 33, 33, 33, 100, 128, 160, 142, 0, 0, 70, 100, 150, 210, 170, 100, 153, 153, 153, 128, 128, 128, 0, 0, 80, 250, 170               , 30, 192, 192, 192, 220, 220, 0, 140, 140, 20, 119, 11, 32, 150, 0, 255, 0, 60, 100, 0, 0, 142, 0, 0, 90, 0, 0, 230, 0, 80, 100, 128, 64, 64, 0, 0, 110, 0, 0, 70, 0, 0                , 192, 32, 32, 32, 120, 10, 10, 0, 0, 0]

zero_pad = 256 * 3 - len(cityscapes_palette)
for i in range(zero_pad):
    cityscapes_palette.append(0)

    
zero_pad = 256 * 3 - len(camvid_palette)
for i in range(zero_pad):
     camvid_palette.append(0)

#zero_pad = 256 * 3 - len(mvd_palette)
#for i in range(zero_pad):
#    mvd_palette.append(0)

def cityscapes_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cityscapes_palette)

    return new_mask


def camvid_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(camvid_palette)

    return new_mask


def mvd_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(mvd_palette)

    return new_mask

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = voc_color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image


def voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap
