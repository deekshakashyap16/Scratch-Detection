import os
import cv2

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

def denoise_image(img, kernel_size=5):
    """
    Removes salt-and-pepper noise using median filtering.
    """
    return cv2.medianBlur(img, kernel_size)

def denoise_flat_folder(input_dir, output_dir, kernel_size=5):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            in_path = os.path.join(input_dir, file)
            out_path = os.path.join(output_dir, file)

            img = cv2.imread(in_path)
            if img is None:
                print(f"Skipping unreadable file: {in_path}")
                continue

            denoised = denoise_image(img, kernel_size)
            cv2.imwrite(out_path, denoised)

    print(f"Denoised images saved to: {output_dir}")

if __name__ == "__main__":
    INPUT_DIR = r"C:\Users\Lenovo\OneDrive\Desktop\Internship\4Good\Internship Test\ScratchDetection\dataset\car_dent_coco\car_dent_coco\train"
    OUTPUT_DIR = "dataset/car_dent_coco/train_denoised"

    denoise_flat_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        kernel_size=5
    )
