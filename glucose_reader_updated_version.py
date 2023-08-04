import cv2
import pytesseract
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilename

def file_upload():
  """
  This function allows users to upload a file from their local system.
  """
  root = tk.Tk()
  root.withdraw()
  filename = askopenfilename()
  return filename

imageType = file_upload()
cv2.waitKey(0)
image = cv2.imread(imageType)
img = plt.imread(imageType)
plt.imshow(img)

#recognizes image
image1 = cv2.imread(imageType)
#converts to grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#applies threshold
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
# Gaussian Blur (9, 75, 75)
Gaussian3 = cv2.bilateralFilter(thresh3, 9, 75, 79)

# Convert the image to a NumPy array
Gaussian3_np = np.array(Gaussian3)

# Display the image
cv2.imshow('Gaussian3', Gaussian3_np)

# cv2.imshow(Gaussian3)
# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# Number:
number3 = pytesseract.image_to_string(Gaussian3, lang='eng', config='--psm 7 outputbase digits tessedit_char_whitelist 0123456789')
print("============",number3)


# waiting until key press
cv2.waitKey()
# destroy all the windows
cv2.destroyAllWindows()

if number3 is not None:
  if int(number3) <= 99:
    print("Thank you for taking the test! All is good!")
  elif int(number3) > 99 and int(number3) < 125:
    print("Thank you for taking the test! Your results show that you have prediabetes. Bubble Health recommends that you see a doctor for further consultation.")
  elif int(number3) >= 126:
    print("Thank you for taking the test! Your results show that you have diabetes. Bubble Health recommends that you see a doctor for further consultation.")
  else:
    print("Please enter another image")
else:
  print("Error: Unable to recognize text from image.")
