
#Extracts plain text out of an uploaded resume file (PDF or DOCX).
import io
import pdfplumber
import docx


class UnsupportedFileType(Exception):
    pass

def extract_text(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return extract_from_pdf(file_bytes)
    elif lower.endswith(".docx"):
        return extract_from_docx(file_bytes)
    else:
        raise UnsupportedFileType(
            f"Unsupported file type for '{filename}'. Please upload a .pdf or .docx file."
        )

def extract_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    file=io.BytesIO(file_bytes)
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_from_docx(file_bytes: bytes) -> str:
    file=io.BytesIO(file_bytes)
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)
