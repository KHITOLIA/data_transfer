import PyPDF2 as py

def decrypt_pdf(input_pdf, output_pdf, password):
    pdf_reader = py.PdfReader(input_pdf)
    pdf_reader.decrypt(password)

    pdf_writer = py.PdfWriter(pdf_reader)
    for page_num in range(len(pdf_reader.pages)):
        pdf_writer.add_page(pdf_reader.pages[page_num])
    
    with open(output_pdf, 'wb') as out:
        pdf_writer.write(out)

    print(f"Decrypted pdf file saved at {output_pdf}")

decrypt_pdf('password_protected.pdf', "removed_password.pdf", "Tushar@2000")