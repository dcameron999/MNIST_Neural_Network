import csv
import cv2
import matplotlib.pyplot as plt


FILE_TO_EXPORT_TO = 'my_number2.csv'
FILE_TO_IMPORT = 'eight2.bmp'
LABEL = 8

# Load image
image_to_import = cv2.imread(FILE_TO_IMPORT, cv2.IMREAD_GRAYSCALE)

# Preview sample image
plt.imshow(image_to_import, cmap='gray')

# Format Image
img_resized = cv2.resize(image_to_import, (28, 28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)

# Preview reformatted image
plt.imshow(img_resized, cmap='gray')
plt.show()

img_resized = img_resized.reshape(1,784)
img_resized_list = img_resized[0].tolist()
img_resized_list.insert(0,LABEL)
print(img_resized_list)

f = open(FILE_TO_EXPORT_TO, 'a',newline='')
w = csv.writer(f, delimiter = ',')
w.writerow(img_resized_list)
f.close()