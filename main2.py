from mobileNet import compare_images

# Example usage:
img1_path = 'agh1_2.jpg'
img2_path = 'agh1_1.jpg'

similarity = compare_images(img1_path, img2_path)
print(f"Similarity: {similarity:.2f}%")
