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

pascal_PALETTE = [120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50,
               4, 200, 3, 120, 120, 80, 140, 140, 140, 204, 5, 255,
               230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7,
               150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82,
               143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3,
               0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255,
               255, 7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220,
               255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224,
               255, 184, 6, 10, 255, 71, 255, 41, 10, 7, 255, 255,
               224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7,
               255, 122, 8, 0, 255, 20, 255, 8, 41, 255, 5, 153,
               6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255,
               140, 140, 140, 250, 10, 15, 20, 255, 0, 31, 255, 0,
               255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255,
               255, 71, 0, 0, 235, 255, 0, 173, 255, 31, 0, 255]

ade_PALETTE = [120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50,
               4, 200, 3, 120, 120, 80, 140, 140, 140, 204, 5, 255,
               230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7,
               150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82,
               143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3,
               0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255,
               255, 7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220,
               255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224,
               255, 184, 6, 10, 255, 71, 255, 41, 10, 7, 255, 255,
               224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7,
               255, 122, 8, 0, 255, 20, 255, 8, 41, 255, 5, 153,
               6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255,
               140, 140, 140, 250, 10, 15, 20, 255, 0, 31, 255, 0,
               255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255,
               255, 71, 0, 0, 235, 255, 0, 173, 255, 31, 0, 255,
               11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255,
               0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163, 0,
               255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0,
               0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255,
               173, 255, 0, 0, 255, 153, 255, 92, 0, 255, 0, 255,
               255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20,
               255, 184, 184, 0, 31, 255, 0, 255, 61, 0, 71, 255,
               255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255,
               0, 112, 255, 51, 0, 255, 0, 194, 255, 0, 122, 255,
               0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0,
               143, 255, 0, 82, 0, 255, 163, 255, 0, 255, 235, 0,
               8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255,
               255, 0, 31, 0, 184, 255, 0, 214, 255, 255, 0, 112,
               92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160,
               163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0, 163,
               255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0,
               255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0,
               10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255,
               255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204,
               41, 0, 255, 41, 255, 0, 173, 0, 255, 0, 245, 255,
               71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255,
               184, 255, 0, 0, 133, 255, 255, 214, 0, 25, 194, 194,
               102, 255, 0, 92, 0, 255]

mvd_palette = [165, 42, 42, 0, 192, 0, 196, 196, 196, 190, 153, 153, 180, 165, 180, 90, 120, 150, 102, 102, 156, 128, 64, 255, 140, 140, 200, 170, 170, 170, 250, 170, 160, 96, 96, 96,                230, 150, 140, 128, 64, 128, 110, 110, 110, 244, 35, 232, 150, 100, 100, 70, 70, 70, 150, 120, 90, 220, 20, 60, 255, 0, 0, 255, 0, 100, 255, 0, 200, 200, 128, 128,                     255, 255, 255, 64, 170, 64, 230, 160, 50, 70, 130, 180, 190, 255, 255, 152, 251, 152, 107, 142, 35, 0, 170, 30, 255, 255, 128, 250, 0, 30, 100, 140, 180, 220, 220, 220,                220, 128, 128, 222, 40, 40, 100, 170, 30, 40, 40, 40, 33, 33, 33, 100, 128, 160, 142, 0, 0, 70, 100, 150, 210, 170, 100, 153, 153, 153, 128, 128, 128, 0, 0, 80, 250, 170               , 30, 192, 192, 192, 220, 220, 0, 140, 140, 20, 119, 11, 32, 150, 0, 255, 0, 60, 100, 0, 0, 142, 0, 0, 90, 0, 0, 230, 0, 80, 100, 128, 64, 64, 0, 0, 110, 0, 0, 70, 0, 0                , 192, 32, 32, 32, 120, 10, 10, 0, 0, 0]

zero_pad = 256 * 3 - len(pascal_PALETTE)
for i in range(zero_pad):
    pascal_PALETTE.append(0)

zero_pad = 256 * 3 - len(ade_PALETTE)
for i in range(zero_pad):
    ade_PALETTE.append(0)

zero_pad = 256 * 3 - len(cityscapes_palette)
for i in range(zero_pad):
    cityscapes_palette.append(0)

    
zero_pad = 256 * 3 - len(camvid_palette)
for i in range(zero_pad):
     camvid_palette.append(0)

#zero_pad = 256 * 3 - len(mvd_palette)
#for i in range(zero_pad):
#    mvd_palette.append(0)

def pas_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(pascal_PALETTE)

    return new_mask

def ade_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(ade_PALETTE)

    return new_mask

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
