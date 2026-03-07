from pypdf import PdfReader
reader = PdfReader(r"D:\Download\Manual-G-TOP-Rev-10-14.09.2021.pdf")
paginas = [page.extract_text() for page in reader.pages]
print(f"{paginas[0]}")


