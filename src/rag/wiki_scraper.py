"""Wiki Scraper - Scrapes the Tower Wiki and ingests into knowledge base."""

import requests
import re
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from .knowledge_base import TowerKnowledgeBase


class WikiScraper:
    """Scrapes the Tower Wiki from Fandom."""

    BASE_URL = "https://the-tower-idle-tower-defense.fandom.com"
    USER_AGENT = "TowerNews/1.0 (Wiki Scraper)"

    def __init__(self):
        self.kb = TowerKnowledgeBase()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        self.last_request = 0
        self.min_delay = 1.0  # Be nice to the wiki

    def _rate_limit(self):
        """Ensure we don't hammer the wiki."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request = time.time()

    def _clean_text(self, text: str) -> str:
        """Clean wiki text for embedding."""
        if not text:
            return ""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove reference markers like [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        return text.strip()

    def get_all_pages(self) -> List[str]:
        """Get list of all wiki pages."""
        self._rate_limit()

        pages = []
        url = f"{self.BASE_URL}/wiki/Special:AllPages"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all page links in the AllPages listing
            content = soup.find('div', class_='mw-allpages-body')
            if content:
                for link in content.find_all('a'):
                    href = link.get('href', '')
                    if href.startswith('/wiki/') and ':' not in href:
                        pages.append(href)

            print(f"[WikiScraper] Found {len(pages)} wiki pages")
            return pages

        except Exception as e:
            print(f"[WikiScraper] Error getting page list: {e}")
            return []

    def get_category_pages(self, category: str) -> List[str]:
        """Get pages from a specific category."""
        self._rate_limit()

        pages = []
        url = f"{self.BASE_URL}/wiki/Category:{category}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find category members
            for link in soup.select('.category-page__member-link'):
                href = link.get('href', '')
                if href.startswith('/wiki/'):
                    pages.append(href)

            return pages

        except Exception as e:
            print(f"[WikiScraper] Error getting category {category}: {e}")
            return []

    def scrape_page(self, page_path: str) -> Dict[str, Any]:
        """Scrape a single wiki page."""
        self._rate_limit()

        url = f"{self.BASE_URL}{page_path}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get title
            title_elem = soup.find('h1', class_='page-header__title')
            title = title_elem.get_text(strip=True) if title_elem else page_path.split('/')[-1]

            # Get main content
            content_div = soup.find('div', class_='mw-parser-output')
            if not content_div:
                return None

            # Extract text content, excluding navigation/infobox clutter
            paragraphs = []
            for elem in content_div.find_all(['p', 'li', 'h2', 'h3']):
                # Skip elements inside tables/infoboxes
                if elem.find_parent('table'):
                    continue
                if elem.find_parent('aside'):
                    continue

                text = elem.get_text(strip=True)
                if len(text) > 20:  # Skip very short elements
                    if elem.name in ['h2', 'h3']:
                        paragraphs.append(f"\n## {text}\n")
                    else:
                        paragraphs.append(text)

            content = '\n'.join(paragraphs)
            content = self._clean_text(content)

            if len(content) < 50:
                return None

            return {
                "title": title,
                "url": url,
                "content": content,
                "page_path": page_path
            }

        except Exception as e:
            print(f"[WikiScraper] Error scraping {page_path}: {e}")
            return None

    def ingest_page(self, page_data: Dict[str, Any]) -> int:
        """Ingest a wiki page into the knowledge base."""
        if not page_data:
            return 0

        title = page_data["title"]
        content = page_data["content"]
        url = page_data["url"]

        # Create document ID from URL
        doc_id = f"wiki_{page_data['page_path'].replace('/', '_')}"

        # Check if already exists
        if self.kb.document_exists(doc_id):
            return 0

        # Chunk long content
        chunks = self._chunk_text(content, max_length=1500)
        added = 0

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk{i}" if len(chunks) > 1 else doc_id

            # Format with title for context
            formatted_content = f"[Wiki: {title}]\n{chunk}"

            result = self.kb.add_document(
                content=formatted_content,
                post_id=chunk_id,
                post_type="wiki",
                metadata={
                    "title": title,
                    "url": url,
                    "source": "fandom_wiki",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                },
                score=500  # High score for wiki content
            )

            if result:
                added += 1

        return added

    def _chunk_text(self, text: str, max_length: int = 1500) -> List[str]:
        """Split long text into chunks."""
        if len(text) <= max_length:
            return [text] if text else []

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def ingest_wiki(self, max_pages: int = 500) -> Dict[str, int]:
        """Ingest the entire wiki into knowledge base."""
        print("[WikiScraper] Starting wiki ingestion...")

        stats = {
            "pages_found": 0,
            "pages_scraped": 0,
            "documents_added": 0
        }

        # Get all pages
        pages = self.get_all_pages()
        stats["pages_found"] = len(pages)

        # Also try to get pages from important categories
        important_categories = [
            "Game_Mechanics",
            "Upgrades",
            "Cards",
            "Modules",
            "Ultimate_Weapons",
            "Labs",
            "Perks",
            "Enemies",
            "Tips_and_Strategies"
        ]

        for category in important_categories:
            cat_pages = self.get_category_pages(category)
            for page in cat_pages:
                if page not in pages:
                    pages.append(page)

        print(f"[WikiScraper] Total unique pages to process: {len(pages)}")

        # Process pages
        for i, page_path in enumerate(pages[:max_pages]):
            if i % 10 == 0:
                print(f"[WikiScraper] Progress: {i}/{min(len(pages), max_pages)}")

            page_data = self.scrape_page(page_path)
            if page_data:
                added = self.ingest_page(page_data)
                if added > 0:
                    stats["pages_scraped"] += 1
                    stats["documents_added"] += added
                    safe_title = page_data['title'][:40].encode('ascii', 'replace').decode('ascii')
                    print(f"[WikiScraper] Added {added} docs from: {safe_title}")

        print(f"[WikiScraper] Ingestion complete:")
        print(f"  Pages found: {stats['pages_found']}")
        print(f"  Pages scraped: {stats['pages_scraped']}")
        print(f"  Documents added: {stats['documents_added']}")

        return stats
