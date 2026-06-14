import PyPDF2 as py
# create password protected pdf files

def encrypt_pdf(input_pdf, output_pdf, password):
    pdf_reader = py.PdfReader(input_pdf)
    pdf_writer = py.PdfWriter(pdf_reader)

    for page_num in range(len(pdf_reader.pages)):
        pdf_writer.add_page(pdf_reader.pages[page_num])

    pdf_writer.encrypt(password)

    with open(output_pdf, 'wb') as out:
        pdf_writer.write(out)
    
    print(f"Encrpyted Pdf Saved at {output_pdf} ")

# example
encrypt_pdf("caste_certificate.pdf", 'password_protected.pdf', 'Tushar@2000')