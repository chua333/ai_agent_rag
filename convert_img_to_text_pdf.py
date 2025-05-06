import os
from pdf2image import convert_from_path
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def convert_images_to_text_pdf(input_dir, output_dir):
    """
    Converts all image-based or scanned PDFs in the input_dir
    to text-based PDFs using OCR, and saves them in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            input_pdf_path = os.path.join(input_dir, filename)
            output_pdf_path = os.path.join(output_dir, f"ocr_{filename}")

            print(f"\n📄 Processing {filename}...")

            try:
                images = convert_from_path(input_pdf_path, dpi=300)

                c = canvas.Canvas(output_pdf_path, pagesize=A4)
                width, height = A4

                for i, image in enumerate(images):
                    print(f" OCR page {i + 1}...")
                    text = pytesseract.image_to_string(image, lang='eng')

                    text_lines = text.split("\n")
                    y_position = height - 50

                    for line in text_lines:
                        if y_position < 50:
                            c.showPage()
                            y_position = height - 50
                        c.drawString(50, y_position, line)
                        y_position -= 14

                    c.showPage()

                c.save()
                print(f"Saved OCR'd PDF to: {output_pdf_path}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# Example usage
input_folder = "documentations"
output_folder = "documentations_ocr"

convert_images_to_text_pdf(input_folder, output_folder)
