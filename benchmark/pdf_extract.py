"""
Le-Route: PDF extraction module.
Extracts text and images separately with page positions.
- Selectable text → extracted as structured text
- Embedded images → extracted as images with bbox positions
- OCR only when text is not machine-readable
"""

import fitz  # pymupdf
import base64
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ExtractedImage:
    """An image extracted from a PDF page."""
    image_b64: str                    # Base64 encoded image
    image_type: str                   # "png", "jpeg", etc.
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) on page
    page_num: int                     # 1-indexed page number
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class ExtractedPage:
    """Content extracted from a single PDF page."""
    page_num: int                     # 1-indexed
    text: str                         # All selectable text on page
    text_blocks: list[dict] = field(default_factory=list)  # Text with positions
    images: list[ExtractedImage] = field(default_factory=list)
    needs_ocr: bool = False           # True if page appears scanned


@dataclass 
class ExtractedDocument:
    """Full document extraction result."""
    pages: list[ExtractedPage]
    total_pages: int
    total_images: int
    total_text_length: int
    

def extract_pdf(pdf_path: str, extract_images: bool = True, min_image_size: int = 0) -> ExtractedDocument:
    """
    Extract text and images from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        extract_images: Whether to extract embedded images
        
    Returns:
        ExtractedDocument with all pages, text, and images
    """
    doc = fitz.open(pdf_path)
    pages = []
    total_images = 0
    total_text_length = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. Extract all selectable text
        text = page.get_text()
        total_text_length += len(text)
        
        # 2. Extract text blocks with positions
        text_blocks = []
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] == 0:  # Type 0 = text block
                block_text = block[4].strip()
                if block_text:
                    text_blocks.append({
                        "text": block_text,
                        "bbox": (block[0], block[1], block[2], block[3]),  # x0, y0, x1, y1
                    })
        
        # 3. Check if OCR might be needed
        needs_ocr = False
        if not text.strip() and page.get_pixmap().samples:
            # Page has pixels but no text - likely scanned
            needs_ocr = True
        
        # 4. Extract images with positions
        images = []
        if extract_images:
            image_list = page.get_images(full=True)
            
            for img_info in image_list:
                xref = img_info[0]
                
                try:
                    # Extract image bytes
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    image_ext = base_image["ext"]
                    
                    # Get image position on page
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        bbox = (
                            img_rects[0].x0,
                            img_rects[0].y0,
                            img_rects[0].x1,
                            img_rects[0].y1
                        )
                    else:
                        bbox = (0, 0, 0, 0)
                    
                    # Filter by minimum size
                    img_width = base_image.get("width", 0)
                    img_height = base_image.get("height", 0)
                    
                    if img_width >= min_image_size and img_height >= min_image_size:
                        images.append(ExtractedImage(
                            image_b64=image_b64,
                            image_type=image_ext,
                            bbox=bbox,
                            page_num=page_num + 1,
                            width=img_width,
                            height=img_height
                        ))
                        total_images += 1
                    
                except Exception as e:
                    print(f"Warning: Could not extract image xref={xref} on page {page_num + 1}: {e}")
        
        pages.append(ExtractedPage(
            page_num=page_num + 1,
            text=text,
            text_blocks=text_blocks,
            images=images,
            needs_ocr=needs_ocr
        ))
    
    doc.close()
    
    return ExtractedDocument(
        pages=pages,
        total_pages=len(pages),
        total_images=total_images,
        total_text_length=total_text_length
    )


def build_mistral_content(
    doc: ExtractedDocument,
    question: str,
    include_images: bool = True,
    max_images: int = 10
) -> list[dict]:
    """
    Build content array for Mistral API (multimodal format).
    
    Args:
        doc: Extracted document
        question: User's question
        include_images: Whether to include images in the request
        max_images: Maximum number of images to include (to manage costs)
        
    Returns:
        List of content blocks for Mistral chat API
    """
    content = []
    
    # 1. Add document text
    text_parts = []
    for page in doc.pages:
        if page.text.strip():
            text_parts.append(f"[Page {page.page_num}]\n{page.text.strip()}")
    
    if text_parts:
        content.append({
            "type": "text",
            "text": "## Document Content:\n\n" + "\n\n---\n\n".join(text_parts)
        })
    
    # 2. Add images with page annotations
    image_count = 0
    if include_images:
        for page in doc.pages:
            for img in page.images:
                if image_count >= max_images:
                    break
                
                # Add page annotation before image
                content.append({
                    "type": "text",
                    "text": f"[Figure from page {img.page_num}]"
                })
                
                # Add the image
                mime_type = f"image/{img.image_type}"
                if img.image_type == "jpg":
                    mime_type = "image/jpeg"
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img.image_b64}"
                    }
                })
                image_count += 1
            
            if image_count >= max_images:
                break
    
    # 3. Add question
    image_note = f" ({image_count} figure(s) included above)" if image_count > 0 else ""
    
    content.append({
        "type": "text",
        "text": f"## Question:\n{question}{image_note}\n\nAnswer based on the document. Cite specific pages."
    })
    
    return content


def get_document_summary(doc: ExtractedDocument) -> dict:
    """Get a summary of the extracted document."""
    return {
        "total_pages": doc.total_pages,
        "total_images": doc.total_images,
        "total_text_chars": doc.total_text_length,
        "pages_needing_ocr": sum(1 for p in doc.pages if p.needs_ocr),
        "images_per_page": [len(p.images) for p in doc.pages]
    }


def save_extracted_images(doc: ExtractedDocument, output_dir: str) -> list[str]:
    """
    Save all extracted images to disk for inspection.
    
    Args:
        doc: Extracted document
        output_dir: Directory to save images
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for page in doc.pages:
        for i, img in enumerate(page.images):
            filename = f"page{img.page_num}_img{i+1}.{img.image_type}"
            filepath = output_path / filename
            
            image_bytes = base64.b64decode(img.image_b64)
            filepath.write_bytes(image_bytes)
            saved_files.append(str(filepath))
    
    return saved_files


# ============== Example Usage ==============

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extract.py <pdf_path> [--save-images <output_dir>]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    save_images = "--save-images" in sys.argv
    output_dir = sys.argv[sys.argv.index("--save-images") + 1] if save_images else None
    
    print(f"Extracting: {pdf_path}")
    
    doc = extract_pdf(pdf_path)
    summary = get_document_summary(doc)
    
    print(f"\n=== Document Summary ===")
    print(f"Pages: {summary['total_pages']}")
    print(f"Total images: {summary['total_images']}")
    print(f"Text length: {summary['total_text_chars']} chars")
    print(f"Pages needing OCR: {summary['pages_needing_ocr']}")
    
    print(f"\n=== Per-Page Details ===")
    for page in doc.pages:
        print(f"\nPage {page.page_num}:")
        print(f"  Text: {len(page.text)} chars")
        print(f"  Images: {len(page.images)}")
        for img in page.images:
            print(f"    - {img.image_type} {img.width}x{img.height} at bbox {tuple(int(x) for x in img.bbox)}")
    
    if save_images and output_dir:
        saved = save_extracted_images(doc, output_dir)
        print(f"\n=== Saved {len(saved)} images to {output_dir} ===")
        for f in saved:
            print(f"  - {f}")
