import PyPDF2
import os

def merge_pdf(pdf_list, output_path):
    """
    Merges multiple PDF files into a single PDF file.

    Parameters:
    pdf_list (list): List of paths to PDF files to be merged.
    output_path (str): Path where the merged PDF will be saved.
    """
    pdf_writer = PyPDF2.PdfWriter()

    for pdf in pdf_list:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)

    with open(output_path, 'wb') as out:
        pdf_writer.write(out)

def pdf_list_from_folder(folder_path):
    pdf_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            pdf_list.append(file)
    return pdf_list

pdf_list = pdf_list_from_folder("D:\live_Projects\pdf")
# print(pdf_list)
merge_pdf(pdf_list, 'merged_output.pdf')
