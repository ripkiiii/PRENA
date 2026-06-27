"""
PRENA Real Data Scraper v2 — new sources to hit 300 real pairs.
Sources: Antara (new tags) + Kontan Press Release + CNBC Indonesia
Appends to prena_real.jsonl, skipping duplicates.
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

NON_PR_KEYWORDS = [
    'piala dunia', 'susunan pemain', 'nilai tukar rupiah', 'kurs dolar',
    'rekomendasi saham', 'pergerakan ihsg', 'sabu', 'tersangka',
    'kriminal', 'demo ', 'mahasiswa demo', 'nonton bareng',
    'jadwal pertandingan', 'live streaming', 'cuaca', 'gempa',
    'banjir', 'kebakaran', 'kecelakaan', 'meninggal',
]


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


def is_non_pr(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in NON_PR_KEYWORDS)


def to_chatml(instruction: str, response: str, category: str, source: str) -> dict:
    return {
        "category": category,
        "source": source,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
    }


# ── Antara new topic tags ─────────────────────────────────────────────
ANTARA_NEW_TAGS = [
    "/tag/investasi",
    "/tag/startup",
    "/tag/pln",
    "/tag/garuda-indonesia",
    "/tag/merger",
    "/tag/akuisisi",
    "/tag/ekspor",
    "/tag/perusahaan",
]


def scrape_antara_new_tags(existing: set) -> list[dict]:
    results = []
    seen_urls = set()

    for tag in ANTARA_NEW_TAGS:
        try:
            r = requests.get(f"https://www.antaranews.com{tag}", headers=HEADERS, timeout=15)
            if r.status_code != 200:
                print(f"  [skip] Antara {tag}: {r.status_code}")
                continue
            soup = BeautifulSoup(r.text, "lxml")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if "antaranews.com/berita/" in a.get("href", "")
                and a["href"] not in seen_urls
            ]
            new = list(dict.fromkeys(links))
            seen_urls.update(new)
            print(f"  Antara {tag}: {len(new)} articles")
            time.sleep(1)
        except Exception as e:
            print(f"  [error] {tag}: {e}")

    # Filter recent (ID > 5_000_000 ≈ 2024+)
    recent = [u for u in seen_urls if (m := re.search(r'/berita/(\d+)/', u)) and int(m.group(1)) > 5_000_000]
    print(f"  → {len(recent)} recent URLs")

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
            if instruction in existing or is_non_pr(title_text):
                continue

            results.append(to_chatml(instruction, content_text[:2000], "press_release", "antaranews.com"))
            existing.add(instruction)
            print(f"  ✓ Antara: {title_text[:70]}")
            time.sleep(1)

        except Exception as e:
            print(f"  [skip] {e}")

    return results


# ── Kontan Press Release ──────────────────────────────────────────────
def scrape_kontan(existing: set) -> list[dict]:
    results = []

    try:
        r = requests.get("https://pressrelease.kontan.co.id/", headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  [skip] Kontan: {r.status_code}")
            return []
        soup = BeautifulSoup(r.text, "lxml")
        links = list(dict.fromkeys(
            a["href"] for a in soup.find_all("a", href=True)
            if "pressrelease.kontan.co.id/news/" in a.get("href", "")
        ))
        print(f"  Kontan: {len(links)} press release links")
    except Exception as e:
        print(f"  [error] Kontan: {e}")
        return []

    for href in links:
        try:
            detail = requests.get(href, headers=HEADERS, timeout=15)
            dsoup = BeautifulSoup(detail.text, "lxml")

            title_el = dsoup.select_one("h1")
            content_el = dsoup.select_one("div.tmpl-article") or dsoup.select_one("div.content-detail")

            if not title_el or not content_el:
                continue

            for tag in content_el.find_all(["script", "style", "figure", "img", "aside"]):
                tag.decompose()

            title_text = clean_text(title_el.get_text())
            content_text = clean_text(content_el.get_text(separator=" "))

            if len(content_text) < 300 or not title_text:
                continue

            instruction = f"Buatkan press release perusahaan tentang: {title_text}"
            if instruction in existing or is_non_pr(title_text):
                continue

            results.append(to_chatml(instruction, content_text[:2000], "press_release", "kontan.co.id"))
            existing.add(instruction)
            print(f"  ✓ Kontan: {title_text[:70]}")
            time.sleep(1)

        except Exception as e:
            print(f"  [skip] {e}")

    return results


# ── CNBC Indonesia ────────────────────────────────────────────────────
def scrape_cnbc(existing: set) -> list[dict]:
    results = []
    sections = [
        "https://www.cnbcindonesia.com/news",
        "https://www.cnbcindonesia.com/market",
    ]

    article_urls = []
    for base_url in sections:
        try:
            r = requests.get(base_url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                print(f"  [skip] CNBC {base_url}: {r.status_code}")
                continue
            soup = BeautifulSoup(r.text, "lxml")
            links = list(dict.fromkeys(
                a["href"] for a in soup.find_all("a", href=True)
                if "cnbcindonesia.com/" in a.get("href", "")
                and "/read/" in a.get("href", "")
            ))
            article_urls.extend(links)
            print(f"  CNBC {base_url.split('/')[-1]}: {len(links)} articles")
        except Exception as e:
            print(f"  [error] CNBC {base_url}: {e}")

    for href in list(dict.fromkeys(article_urls))[:50]:
        try:
            detail = requests.get(href, headers=HEADERS, timeout=15)
            dsoup = BeautifulSoup(detail.text, "lxml")

            title_el = dsoup.select_one("h1")
            content_el = dsoup.select_one("div.detail_text") or dsoup.select_one("article")

            if not title_el or not content_el:
                continue

            for tag in content_el.find_all(["script", "style", "figure", "img", "aside"]):
                tag.decompose()

            title_text = clean_text(title_el.get_text())
            content_text = clean_text(content_el.get_text(separator=" "))
            content_text = re.sub(r'^Jakarta,\s*CNBC\s*Indonesia\s*[-–]\s*', '', content_text)

            if len(content_text) < 300 or not title_text:
                continue

            instruction = f"Tuliskan berita atau press release korporasi tentang: {title_text}"
            if instruction in existing or is_non_pr(title_text):
                continue

            results.append(to_chatml(instruction, content_text[:2000], "press_release", "cnbcindonesia.com"))
            existing.add(instruction)
            print(f"  ✓ CNBC: {title_text[:70]}")
            time.sleep(1)

        except Exception as e:
            print(f"  [skip] {e}")

    return results


def main():
    existing = load_existing()
    print(f"Existing pairs: {len(existing)} — target: 300 real pairs\n")

    all_new = []

    print("[ANTARA NEW TAGS] investasi/startup/pln/garuda/merger/ekspor...")
    antara = scrape_antara_new_tags(existing)
    all_new.extend(antara)
    print(f"  → {len(antara)} new pairs\n")

    print("[KONTAN PRESS RELEASE] pressrelease.kontan.co.id...")
    kontan = scrape_kontan(existing)
    all_new.extend(kontan)
    print(f"  → {len(kontan)} new pairs\n")

    print("[CNBC INDONESIA] news + market...")
    cnbc = scrape_cnbc(existing)
    all_new.extend(cnbc)
    print(f"  → {len(cnbc)} new pairs\n")

    if all_new:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for entry in all_new:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Appended {len(all_new)} new pairs → total real: {len(existing)} pairs")
    else:
        print("No new pairs found.")


if __name__ == "__main__":
    main()
