from services.vlm_attributes_service import VLMAttributesService
import cv2


service = VLMAttributesService()

image_path = r"gallery\candidates\candidate_2.webp"

image = cv2.imread(image_path)

if image is None:
    print("Image load failed")
    raise SystemExit


prompt = "Are the person's hair visible in the image? Answer only YES or NO."

result = service.extract_attributes(
    image,
    prompt
)

print("\nVLM RESULT")
print(result)