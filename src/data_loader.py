# data_loader.py - Load and clean the 20 Newsgroups corpus.



import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from src.config import NEWSGROUPS_PATH, USE_MINI, DATA_DIR


# Header fields to strip completely
HEADER_FIELDS_TO_STRIP = {
    "from", "path", "newsgroups", "message-id", "date", "references",
    "sender", "organization", "lines", "nntp-posting-host", "x-newsreader",
    "x-mailer", "mime-version", "content-type", "content-transfer-encoding",
    "return-path", "received", "reply-to", "x-xxdate", "x-useragent"
}


def clean_article(raw_text: str) -> str:
    lines = raw_text.split('\n')
    cleaned_lines = []
    in_header = True
    subject_text = ""

    for line in lines:
        stripped = line.strip()

        # Header section ends at first blank line
        if in_header:
            if stripped == "":
                in_header = False
                continue
            # Extract subject as it's topically meaningful
            if stripped.lower().startswith("subject:"):
                subject_text = stripped[8:].strip()
                # Remove RE: markers
                subject_text = re.sub(r'^(re:\s*)+', '', subject_text, flags=re.IGNORECASE).strip()
            # Check if it's a header field we want to skip
            header_key = stripped.split(":")[0].lower()
            if header_key in HEADER_FIELDS_TO_STRIP:
                continue
            continue  # Skip all header lines except subject (added below)

        # Body section
        # Skip quoted reply lines
        if stripped.startswith('>'):
            continue
        # Skip signature separator and everything after
        if stripped == '--' or stripped == '---':
            break
        # Skip lines that are just dashes or equals (common separators)
        if re.match(r'^[-=_]{3,}$', stripped):
            continue
        # Skip lines that look like attribution (e.g. "In article <...> someone wrote:")
        if re.match(r'^In article .* wrote:$', stripped, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    body = '\n'.join(cleaned_lines).strip()

    # Prepend subject to body for richer semantic content
    if subject_text:
        body = subject_text + ". " + body

    # Normalize whitespace
    body = re.sub(r'\n{3,}', '\n\n', body)
    body = re.sub(r'[ \t]+', ' ', body)

    return body.strip()


def load_corpus(
    data_path: Optional[Path] = None,
    min_tokens: int = 50,
    max_docs_per_category: Optional[int] = None
) -> List[Dict]:
    if data_path is None:
        data_path = DATA_DIR / ("mini_newsgroups" if USE_MINI else "20_newsgroups")

    documents = []
    seen_hashes = set()  # for deduplication
    
    categories = [d for d in data_path.iterdir() if d.is_dir()]
    categories.sort()  # deterministic ordering
    
    print(f"Loading corpus from: {data_path}")
    print(f"Found {len(categories)} categories")

    for category_dir in tqdm(categories, desc="Loading categories"):
        category = category_dir.name
        article_files = list(category_dir.iterdir())
        
        if max_docs_per_category:
            article_files = article_files[:max_docs_per_category]

        for article_path in article_files:
            if not article_path.is_file():
                continue
            
            try:
                # Try UTF-8 first, fall back to latin-1 (common in old Usenet)
                try:
                    raw_text = article_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    raw_text = article_path.read_text(encoding='latin-1')

                cleaned = clean_article(raw_text)
                
                # Minimum length filter
                if len(cleaned.split()) < min_tokens:
                    continue
                
                # Deduplication via content hash
                content_hash = hashlib.md5(cleaned.encode()).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                documents.append({
                    "doc_id": f"{category}/{article_path.name}",
                    "category": category,
                    "text": cleaned,
                    "raw_text": raw_text,
                })

            except Exception as e:
                # Silently skip unreadable files — these are typically binary attachments
                continue

    print(f"Loaded {len(documents)} documents after cleaning and deduplication")
    return documents


if __name__ == "__main__":
    docs = load_corpus()
    print(f"Sample doc:\n{docs[0]['text'][:300]}")
    print(f"Category distribution:")
    from collections import Counter
    cats = Counter(d['category'] for d in docs)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")