import requests
import json
import time
import re
from bs4 import BeautifulSoup

OUTPUT_FILE = "../dataset/raw/real/prena_real.jsonl"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

SEEN_URLS = set()  # global dedup across all scrapers


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def to_chatml(instruction: str, response: str, category: str, source: str) -> dict:
    return {
        "category": category,
        "source": source,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
    }


# ── Scraper 1: Antara Rilis Pers ──────────────────────────────────────
# Combines /rilis-pers and /tag/siaran-pers, global dedup
def scrape_antara() -> list[dict]:
    results = []
    antara_paths = ["/rilis-pers", "/tag/siaran-pers"]

    article_urls = []
    for path in antara_paths:
        try:
            url = f"https://www.antaranews.com{path}"
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                print(f"  [skip] Antara {path}: {r.status_code}")
                continue
            soup = BeautifulSoup(r.text, "lxml")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if "antaranews.com/berita/" in a.get("href", "")
                and a["href"] not in SEEN_URLS
            ]
            new_links = list(dict.fromkeys(links))  # preserve order, dedup within path
            article_urls.extend(new_links)
            SEEN_URLS.update(new_links)
            print(f"  Antara {path}: {len(new_links)} unique articles")
        except Exception as e:
            print(f"  [error] Antara {path}: {e}")

    for href in article_urls:
        try:
            detail = requests.get(href, headers=HEADERS, timeout=15)
            dsoup = BeautifulSoup(detail.text, "lxml")

            title = dsoup.select_one("h1")
            content_sec = dsoup.find("section", class_="pb-80")

            if not title or not content_sec:
                continue

            for tag in content_sec.find_all(["script", "style", "nav", "aside", "figure", "img"]):
                tag.decompose()

            title_text = clean_text(title.get_text())
            content_text = clean_text(content_sec.get_text(separator=" "))
            # Strip breadcrumb header noise
            content_text = re.sub(r'^ANTARA\s+.*?WIB\s*waktu baca\s*\d+\s*menit\s*', '', content_text).strip()
            # Strip image captions at start
            content_text = re.sub(r'^\([^)]+\)\s*', '', content_text)

            if len(content_text) < 300 or not title_text:
                continue

            instruction = f"Buatkan rilis pers dalam Bahasa Indonesia tentang: {title_text}"
            results.append(to_chatml(instruction, content_text[:2000], "press_release", "antaranews.com"))
            print(f"  ✓ Antara: {title_text[:70]}")
            time.sleep(1)

        except Exception as e:
            print(f"  [skip] {href[-60:]}: {e}")
            continue

    return results


# ── Scraper 2: Bisnis.com Ekonomi & Finansial ─────────────────────────
# Corporate press-release style articles from BUMN and major companies
def scrape_bisnis() -> list[dict]:
    results = []
    sections = [
        ("https://ekonomi.bisnis.com/", "press_release"),
        ("https://finansial.bisnis.com/", "press_release"),
    ]

    article_urls = []
    for base_url, _ in sections:
        try:
            r = requests.get(base_url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                print(f"  [skip] Bisnis {base_url}: {r.status_code}")
                continue
            soup = BeautifulSoup(r.text, "lxml")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if "bisnis.com/read/" in a.get("href", "")
                and a["href"] not in SEEN_URLS
            ]
            new_links = list(dict.fromkeys(links))
            article_urls.extend(new_links)
            SEEN_URLS.update(new_links)
            print(f"  Bisnis {base_url}: {len(new_links)} unique articles")
        except Exception as e:
            print(f"  [error] Bisnis {base_url}: {e}")

    for href in article_urls[:40]:
        try:
            detail = requests.get(href, headers=HEADERS, timeout=15)
            dsoup = BeautifulSoup(detail.text, "lxml")

            title = dsoup.select_one("h1")
            content_el = dsoup.select_one("article")

            if not title or not content_el:
                continue

            for tag in content_el.find_all(["script", "style", "figure", "img", "aside"]):
                tag.decompose()

            title_text = clean_text(title.get_text())
            content_text = clean_text(content_el.get_text(separator=" "))
            # Strip "Bisnis.com, KOTA -" lead
            content_text = re.sub(r'^Bisnis\.com,\s*[A-Z\s]+-\s*', '', content_text)

            if len(content_text) < 300 or not title_text:
                continue

            instruction = f"Tuliskan press release atau berita korporasi tentang: {title_text}"
            results.append(to_chatml(instruction, content_text[:2000], "press_release", "bisnis.com"))
            print(f"  ✓ Bisnis: {title_text[:70]}")
            time.sleep(1)

        except Exception as e:
            print(f"  [skip] {e}")
            continue

    return results


def main():
    all_results = []

    print("\n[ANTARA] Scraping rilis pers...")
    antara = scrape_antara()
    all_results.extend(antara)
    print(f"  → {len(antara)} pairs dari Antara")

    print("\n[BISNIS.COM] Scraping berita korporasi...")
    bisnis = scrape_bisnis()
    all_results.extend(bisnis)
    print(f"  → {len(bisnis)} pairs dari Bisnis.com")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(all_results)} pairs saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
