import os
import PyPDF2

def split_pdf(pdf_path, output_dir):
    """
    Splits a PDF file into individual pages and saves each page as a separate PDF file.

    :param pdf_path: Path to the input PDF file.
    :param output_dir: Directory where the split PDF files will be saved.
    """
    os.makedirs(output_dir, exist_ok = True)

    # open the pdf file
    pdf_reader = PyPDF2.PdfReader(pdf_path)

    # loop through each page and save it as a separate pdf
    for page_num, page in enumerate(pdf_reader.pages):
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(page)

        # path for output pdf files
        output_file = os.path.join(output_dir, f'page_{page_num}.pdf')

        # write the single page to new file
        with open(output_file, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)

        print(f"saved :{len(pdf_reader.pages)} pages saved in '{output_dir}'.")

if __name__ == "__main__":
    split_pdf("merged_output.pdf", 'pdf_files')