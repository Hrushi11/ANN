import img_processing

# Read the image
image = img_processing.read_image('image.jpg')

# Apply thresholding
thresholded_image = img_processing.threshold(image, threshold=128)

# Apply convolution
convolved_image = img_processing.convolve(thresholded_image, kernel=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Save the output image
img_processing.save_image(convolved_image, 'output_image.jpg')