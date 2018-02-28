import argparse
import scipy.misc
import numpy as np
import sys
import glob
sys.path.append('./')
parser = argparse.ArgumentParser()
parser.add_argument("--src", help="src image path", type=str)
parser.add_argument("--dst", help="dst image path", type=str)
parser.add_argument("--sample_number", help="sample number", type=int)
def process_img_origin(file):
	try:
		t = scipy.misc.imread(file)
	except:
		return None
	return t
if __name__ == '__main__':
	args = parser.parse_args()
	imgs = glob.glob(args.src+'/*.jpeg')
	np.random.shuffle(imgs)
	for i,m in enumerate(imgs[:arg.sample_number]):
		img = process_img_origin(m)
		if img != None and len(img.shape) != 0:
			scipy.misc.imsave(args.dst+'/'+str(i).zfill(6)+'.jpg',img)

