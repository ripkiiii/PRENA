"""
Extra real data scraper for PRENA.
Sources: Antara topic tags (BUMN/korporasi/bank) + Liputan6 bisnis
Appends to prena_real.jsonl, skipping already-scraped articles.
"""
import requests
import json
import time
import re
from bs4 import BeautifulSoup

OUTPUT_FILE = "../dataset/raw/real/prena_real.jsonl"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Load existing instructions to avoid duplicates
def load_existing() -> set:
    seen = set()
    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                seen.add(d["messages"][0]["content"])
    except FileNotFoundError:
        pass
    return seen


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


# ── Antara topic tags ─────────────────────────────────────────────────
ANTARA_TAGS = [
    "/tag/bumn",
    "/tag/korporasi",
    "/tag/bank",
    "/tag/telkom",
    "/tag/pertamina-2",
]

NON_PR_KEYWORDS = [
    'piala dunia', 'susunan pemain', 'nilai tukar rupiah',
    'rekomendasi saham', 'pergerakan ihsg', 'sabu', 'tersangka',
    'kriminal', 'demo ', 'mahasiswa demo', 'nonton bareng',
    'jadwal pertandingan', 'live streaming',
]


def is_non_pr(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in NON_PR_KEYWORDS)


def scrape_antara_tags(existing: set) -> list[dict]:
    results = []
    seen_urls = set()

    # Collect all unique article URLs from topic tags
    for tag in ANTARA_TAGS:
        try:
            r = requests.get(f"https://www.antaranews.com{tag}", headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if "antaranews.com/berita/" in a.get("href", "")
                and a["href"] not in seen_urls
            ]
            new = [u for u in dict.fromkeys(links)]
            seen_urls.update(new)
            print(f"  Antara {tag}: {len(new)} articles")
        except Exception as e:
            print(f"  [error] {tag}: {e}")

    # Filter: recent only (ID > 5_000_000 ≈ 2024+)
    recent = []
    for url in seen_urls:
        m = re.search(r'/berita/(\d+)/', url)
        if m and int(m.group(1)) > 5_000_000:
            recent.append(url)

    print(f"  → {len(recent)} recent URLs to scrape")

    for href in recent:
        try:
            detail = requests.get(href, headers=HEADERS, timeout=15)
            dsoup = BeautifulSoup(detail.text, "lxml")

            title_el = dsoup.select_one("h1")
            content_sec = dsoup.find("section", class_="pb-80")

            if not title_el or not content_sec:
                continue

            for tag in content_sec.find_all(["script", "style", "nav", "aside", "figure", "img"]):
                tag.decompose()

            title_text = clean_text(title_el.get_text())
            content_text = clean_text(content_sec.get_text(separator=" "))
            content_text = re.sub(r'^ANTARA\s+.*?WIB\s*waktu baca\s*\d+\s*menit\s*', '', content_text).strip()
            content_text = re.sub(r'^\([^)]+\)\s*', '', content_text)

            if len(content_text) < 300 or not title_text:
                continue

            instruction = f"Buatkan rilis pers atau siaran pers korporasi tentang: {title_text}"

            # Skip if already in dataset
            if instruction in existing or is_non_pr(title_text):
                continue

            results.append(to_chatml(instruction, content_text[:2000], "press_release", "antaranews.com"))
            existing.add(instruction)
            print(f"  ✓ Antara: {title_text[:70]}")
            time.sleep(1)

        except Exception as e:
            print(f"  [skip] {e}")
            continue

    return results


# ── Liputan6 Bisnis ───────────────────────────────────────────────────
def scrape_liputan6_bisnis(existing: set) -> list[dict]:
    results = []

    try:
        r = requests.get("https://www.liputan6.com/bisnis", headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "lxml")
        links = list(dict.fromkeys(
            a["href"] for a in soup.find_all("a", href=True)
            if "liputan6.com/bisnis/read/" in a.get("href", "")
        ))
        print(f"  Liputan6 /bisnis: {len(links)} article links")
    except Exception as e:
        print(f"  [error] Liputan6: {e}")
        return []

    for href in links:
        try:
            detail = requests.get(href, headers=HEADERS, timeout=15)
            dsoup = BeautifulSoup(detail.text, "lxml")

            title_el = dsoup.select_one("h1")
            content_el = dsoup.select_one("div.article-content-body")

            if not title_el or not content_el:
                continue

            for tag in content_el.find_all(["script", "style", "figure", "img", "aside"]):
                tag.decompose()

            title_text = clean_text(title_el.get_text())
            content_text = clean_text(content_el.get_text(separator=" "))
            # Strip "Liputan6.com, KOTA -" lead
            content_text = re.sub(r'^Liputan6\.com,\s*[A-Z\s]+-\s*', '', content_text)

            if len(content_text) < 300 or not title_text:
                continue

            instruction = f"Buatkan press release atau berita bisnis korporasi tentang: {title_text}"

            if instruction in existing or is_non_pr(title_text):
                continue

            results.append(to_chatml(instruction, content_text[:2000], "press_release", "liputan6.com"))
            existing.add(instruction)
            print(f"  ✓ Liputan6: {title_text[:70]}")
            time.sleep(1)

        except Exception as e:
            print(f"  [skip] {e}")
            continue

    return results


def main():
    existing = load_existing()
    print(f"Existing pairs: {len(existing)}")
    all_new = []

    print("\n[ANTARA TOPIC TAGS] Scraping BUMN/korporasi/bank/telkom/pertamina...")
    antara = scrape_antara_tags(existing)
    all_new.extend(antara)
    print(f"  → {len(antara)} new pairs dari Antara topic tags")

    print("\n[LIPUTAN6] Scraping bisnis section...")
    liputan6 = scrape_liputan6_bisnis(existing)
    all_new.extend(liputan6)
    print(f"  → {len(liputan6)} new pairs dari Liputan6")

    if all_new:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for entry in all_new:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nAppended {len(all_new)} new pairs → total now: {len(existing)} pairs")
    else:
        print("\nNo new pairs found.")


if __name__ == "__main__":
    main()
