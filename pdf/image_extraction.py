import fitz # library that provides functionality for image extraction using pymupdf
import os

def extract_images(pdf_path, images_dir):
    os.makedirs(images_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index)
        image_list = page.get_images(full = True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image['image']
            image_extention = base_image['ext']
            image_filename = f"{images_dir}/image_{page_index + 1}_{img_index + 1}.{image_extention}"

            with open(image_filename, 'wb') as image_file:
                image_file.write(image_bytes)
            print(f"Saved : {image_filename}")
# let's use the above function
# extract_images("caste_certificate.pdf", 'images')

for pdf_file in os.listdir('D:\live_Projects\pdf'):
    if pdf_file.endswith('.pdf'):
        extract_images(pdf_file, 'images')
