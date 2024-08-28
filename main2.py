from mobileNet import compare_images

# Example usage:
img1_path = 'agh1_2.jpg'
img2_path = 'agh1_1.jpg'

similarity_index, threshold_result, confidence_result = compare_images(img1_path, img2_path)
# print(f"Similarity: {similarity:.2f}%")
# print(similarity)

# print(f"Percentage Similarity: {similarity_index:.2f}%")
print(f"Percentage Similarity: {similarity_index:.2f}%")
# print(f"Confidence Value: {confidence_value:.2f}%")
# print(f"Threshold Result (Average Intensity): {threshold_result:.2f}")
print(f"Threshold Result (Average Intensity): {threshold_result:.5f}")
print(f"Confidence Level: {confidence_result:.5f}")