import pdfplumber
import warnings
warnings.filterwarnings("ignore")


def extract_text_from_pdf(pdf_path, output_text_file):
    
    with pdfplumber.open(pdf_path) as pdf:

        text = ''
        for page in pdf.pages:
            page_text = page.extract_text() 

            text += page_text + '\n'

        with open(output_text_file, 'w', encoding="utf-8") as f:
            f.write(text)
        print(f"Text extracted and saved to '{output_text_file}'.")

if __name__ == "__main__":
    extract_text_from_pdf('instruction_notification.pdf', 'text_file.txt')