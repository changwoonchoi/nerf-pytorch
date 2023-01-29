import cv2
import numpy as np

depth = 1 - np.load('/home/ccw/Downloads/105943_depth.npy')
depth_map = cv2.resize(depth, (750, 750))

new_img = np.zeros((1000, 750))
new_img += 0.5
new_img[125:875, :] = depth_map
np.save('/home/ccw/Downloads/105943_depth_filled.npy', new_img)
# new_img = new_img * 255
cv2.imwrite('/home/ccw/Downloads/105943_depth.png', new_img)
breakpoint()

print(1)