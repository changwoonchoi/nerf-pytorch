import cv2
import numpy as np
from sklearn.preprocessing import normalize


def load_image_from_path(image_file_path, scale=1):
	image = cv2.imread(image_file_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if scale != 1:
		image = cv2.resize(image, None, fx=scale, fy=scale)
	image = image.astype(np.float32)
	image /= 255.0

	return image
