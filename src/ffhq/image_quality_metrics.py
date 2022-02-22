from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
from skimage import io
import argparse
import matplotlib as mathplt

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True)
ap.add_argument("-s", "--second", required=True)
args = vars(ap.parse_args())

grayA = io.imread(args["first"], as_gray=True)
grayB = io.imread(args["second"], as_gray=True)

mse = mean_squared_error(grayA, grayB)
ssim = structural_similarity(grayA, grayB)
psnr = peak_signal_noise_ratio(grayA, grayB)

print("MSE: {}".format(mse))
print("SSIM: {}".format(ssim))
print("PSNR: {}".format(psnr))
