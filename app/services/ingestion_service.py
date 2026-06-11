import logging
from typing import List,Dict,Any
from pathlib import Path
import tiktoken

logger = logging.getLogger(f"rag_app.{__name__}")

# Import Docling components
try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    from docling_core.transforms.chunker import DocChunk
    DOCLING_AVAILABLE = True

except Exception as e:
    logger.warning(f"Docling not available - {e}")
    DOCLING_AVAILABLE = False

class DoclingService:
    def __init__(self):
        
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not vailable. please install it")
        
        self.converter = DocumentConverter()
        self.encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
        self.tokenizer = OpenAITokenizer(tokenizer=self.encoding, max_tokens=512)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True
        )

    def convert_document(self, file_path: str):
        """
    Convert document using Docling's advanced layout analysis.

    Args:
        file_path: Path to the document file

    Returns:
        DoclingDocument: Structured document with hierarchy preserved

    Raises:
        ImportError: If Docling is not installed
        Exception: If conversion fails
    """
                       
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not Found: {path.name}")
        
        try:
            logger.info(f"Converting document with Docling: {path.name}")
            result = self.converter.convert(file_path)
            doc = result.document

            logger.info(f"Document converted successfully: {path.name}")
            return doc
        
        except Exception as e:
            logger.error(f"Docling conversion failed: {str(e)}")
            raise

    
    def chunk_document(self, doc) -> List[DocChunk]:
        """
        Chunk a DoclingDocument and return DocChunk objects.
        """
        try:
            logger.info("Starting HybridChunker Processing")
            raw_chunks = list(self.chunker.chunk(dl_doc=doc))
            logger.info(f"generated {len(raw_chunks)} chunks")

            return raw_chunks
        
        except Exception as e:
            logger.error(f"HybridChunking failed: {str(e)}")
            raise
    

    def count_chunk_tokens(self, chunk: DocChunk) ->int:
        """
        Count tokens in a DocChunk.
        """
        text = self.get_chunk_text(chunk)
        return self.tokenizer.count_tokens(text)
    

    def get_chunk_text(self, chunk: DocChunk) ->str:
        """
        Convert a DocChunk to text.
        """
        return self.chunker.contextualize(chunk)
    
    def extract_metadata(self, chunks: List[DocChunk]) ->List[Dict[str,Any]]:
        """
        Extract metadata from Docling chunks for downstream RAG storage.

        Args:
            chunks: List of DocChunk objects

        Returns:
            List of dictionaries containing chunk text + metadata
        """

        extracted=[]
        logger.info(f"Starting Metadata Extraction for {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            try:
                
                # 1. Safely extract core fields
                text = self.get_chunk_text(chunk)
                meta = getattr(chunk, "meta", None)

                chunk_metadata = {
                    "chunk_id": f"chunk_{i}",
                    "text": text,
                    "tokens": self.count_chunk_tokens(chunk),
                    "headings": [],
                    "pages": [],
                    "labels": []
                }

                if meta:
                    # 2. Extract Headings list directly
                    chunk_metadata["headings"]= getattr(meta, "headings", [])
                    # 3. Handle nested doc_items array to extract structural labels and true pages
                    doc_items = getattr(meta, "doc_items", []) or []

                    pages_found = set()
                    labels_found = set()

                    for item in doc_items:
                        # Capture unique content structural labels (e.g., 'text', 'heading', 'table')
                        item_label = getattr(item, "label", None)
                        if item_label:
                            # Convert enum or string safely to text (e.g., "text")
                            labels_found.add(getattr(item_label,"name", str(item_label)).lower())
                        
                        # Dig deep into provenance layout blocks for true page numbers
                        provenance_list = getattr(item, "prov", [])
                        for prov in provenance_list:
                            page_no = getattr(prov, "page_no", None)
                            if page_no is not None:
                                pages_found.add(page_no)

                    # Update collections if data was discovered
                    if pages_found:
                        chunk_metadata["pages"]= sorted(list(pages_found))
                    if labels_found:
                        chunk_metadata["labels"] = sorted(list(labels_found))
                    
                extracted.append(chunk_metadata)
   
                    
            except Exception as e:
                logger.error(f"Metadata Extraction failed: {str(e)}")
                continue

        logger.info(f"Metadata Extraction completed: {len(extracted)} chunks")
        return extracted
