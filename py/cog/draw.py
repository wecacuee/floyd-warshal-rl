import numpy as np


USE_MATPLOTLIB = True

def cv2_white_img(imsize):
    return 255*np.ones((imsize[0], imsize[1], 3), dtype='u1')


if USE_MATPLOTLIB:
    from .mpl2cv2 import MPLAsCV
    backend = MPLAsCV()
else:
    import cv2 as backend
    white_img = cv2_white_img
    to_ndarray = lambda im : im
    from_ndarray = lambda arr : arr

IMPORT_NAMES = """
white_img
circle
fillConvexPoly
line
rectangle
polylines
putText
imshow
imwrite
waitKey
namedWindow""".split()
for n in IMPORT_NAMES:
    globals()[n] = getattr(backend, n)

def color_from_rgb(rgb):
    return rgb[::-1]
