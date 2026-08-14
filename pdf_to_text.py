from pypdf import PdfReader
from pathlib import Path

pdf_path = Path("foundations/me/ProfileLinkedin.pdf")
txt_path = Path("foundations/me/ProfileLinkedin.txt")

reader = PdfReader(pdf_path)

text_parts = []
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text_parts.append(page_text)

txt_path.write_text("\n\n".join(text_parts), encoding="utf-8")

print(f"Saved text to {txt_path}")
