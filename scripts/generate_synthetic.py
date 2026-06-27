import json
import os
import time
import random
from openai import OpenAI

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-r1:free"
OUTPUT_FILE = "../dataset/raw/synthetic/prena_synthetic_v2.jsonl"
PAIRS_PER_CALL = 10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

CATEGORIES = {
    "press_release": {
        "count": 20,
        "topic": "penulisan siaran pers / press release perusahaan atau instansi Indonesia",
        "prompt_extra": """Konteks: perusahaan FMCG, startup tech, BUMN, atau instansi pemerintah Indonesia.
Task bisa berupa: peluncuran produk baru, pencapaian perusahaan, kerjasama strategis, respons isu publik, acara perusahaan.
Instruksi harus menyebutkan konteks spesifik (nama brand fiktif, industri, topik).
Respons harus berformat press release lengkap: judul, lead paragraph, body, kutipan narasumber, boilerplate."""
    },
    "caption_sosmed": {
        "count": 40,
        "topic": "pembuatan caption media sosial untuk brand Indonesia (Instagram, Twitter/X, LinkedIn, TikTok)",
        "prompt_extra": """Konteks: brand lokal Indonesia dari berbagai industri (F&B, fashion, tech, pemerintah, NGO).
Task bisa berupa: caption produk baru, konten campaign, konten informatif, konten engagement, hari nasional.
Instruksi harus menyebutkan platform, tone brand, dan konteks konten.
Respons harus sesuai platform: Instagram (visual + hashtag), Twitter (singkat), LinkedIn (profesional), TikTok (hook kuat)."""
    },
    "crisis_communication": {
        "count": 30,
        "topic": "komunikasi krisis dan manajemen reputasi brand atau instansi Indonesia",
        "prompt_extra": """Konteks: krisis PR yang umum terjadi di Indonesia — produk viral bermasalah, isu karyawan, kebijakan kontroversial, hoaks, kecelakaan, pencemaran lingkungan.
Task bisa berupa: draft pernyataan resmi, respons komentar negatif viral, klarifikasi di media sosial, surat terbuka.
Instruksi harus menyebutkan jenis krisis dan platform/medium respons.
Respons harus empati, jelas, tidak defensif, dan actionable."""
    },
    "media_monitoring": {
        "count": 30,
        "topic": "analisis media monitoring dan sentimen pemberitaan brand atau isu di Indonesia",
        "prompt_extra": """Konteks: laporan media monitoring untuk brand, instansi pemerintah, atau tokoh publik Indonesia.
Task bisa berupa: rangkuman sentimen berita minggu ini, analisis tone media, identifikasi isu yang sedang naik, laporan coverage media.
Instruksi bisa berupa teks berita/komentar yang perlu dianalisis, atau permintaan draft laporan monitoring.
Respons harus terstruktur: sentimen (positif/negatif/netral), poin utama, rekomendasi."""
    },
    "media_pitch": {
        "count": 20,
        "topic": "penulisan media pitch dan email ke jurnalis atau editor media Indonesia",
        "prompt_extra": """Konteks: PR practitioner yang ingin mempitch cerita ke media Indonesia (Kompas, Detik, Tempo, Bisnis Indonesia, Tech in Asia ID, dll).
Task bisa berupa: email pitch liputan, subject line yang menarik, angle cerita untuk jurnalis, follow-up pitch.
Instruksi harus menyebutkan media target, topik pitch, dan angle yang ditawarkan.
Respons harus singkat, hook kuat di awal, news value jelas."""
    },
    "brand_storytelling": {
        "count": 20,
        "topic": "penulisan narasi dan brand storytelling untuk perusahaan atau produk Indonesia",
        "prompt_extra": """Konteks: brand lokal Indonesia yang ingin membangun narasi kuat di era digital.
Task bisa berupa: company profile singkat, origin story brand, narasi untuk about page, tagline + penjelasan, brand manifesto.
Instruksi harus menyebutkan industri, nilai brand, dan target audiens.
Respons harus autentik, emosional, dan mencerminkan identitas lokal Indonesia."""
    },
    "influencer_brief": {
        "count": 10,
        "topic": "pembuatan creative brief untuk kolaborasi dengan influencer Indonesia",
        "prompt_extra": """Konteks: brand Indonesia yang mau kolaborasi dengan influencer/KOL di Instagram, TikTok, atau YouTube.
Task bisa berupa: brief konten untuk influencer, talking points kampanye, do's and don'ts konten, caption template untuk influencer.
Instruksi harus menyebutkan tier influencer (nano/micro/macro), platform, dan produk/campaign.
Respons harus jelas, mudah dipahami kreator, dan menjaga tone brand."""
    },
}

SYSTEM_PROMPT = """Kamu adalah generator dataset instruksi Digital Public Relations (PR) Bahasa Indonesia berkualitas tinggi.

Tugasmu: buat {n} pasang instruksi-respons dalam Bahasa Indonesia tentang: {topic}

{extra}

Output HARUS berupa JSON array seperti ini:
[
  {{
    "instruction": "instruksi atau permintaan yang natural dari seorang PR practitioner",
    "response": "respons profesional dan lengkap dalam Bahasa Indonesia"
  }},
  ...
]

Aturan:
- Instruksi harus natural, seperti yang diminta PR practitioner sungguhan di Indonesia
- Respons harus profesional, praktikal, dan siap pakai
- Variasikan industri, tone, dan konteks di setiap pasang
- Gunakan nama brand/perusahaan FIKTIF tapi realistis (nama Indo)
- HANYA output JSON array, tidak ada teks lain"""


def generate_batch(category: str, config: dict, n: int = PAIRS_PER_CALL) -> list[dict]:
    prompt = SYSTEM_PROMPT.format(
        n=n,
        topic=config["topic"],
        extra=config["prompt_extra"]
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )
    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    pairs = json.loads(content)
    return pairs


def to_chatml(instruction: str, response: str, category: str) -> dict:
    return {
        "category": category,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    total_done = 0
    total_target = sum(c["count"] for c in CATEGORIES.values())

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for category, config in CATEGORIES.items():
            target = config["count"]
            calls = target // PAIRS_PER_CALL
            remainder = target % PAIRS_PER_CALL
            if remainder:
                calls += 1

            cat_done = 0
            print(f"\n[{category.upper()}] target: {target} pairs, {calls} calls")

            for i in range(calls):
                n = PAIRS_PER_CALL if i < calls - 1 or remainder == 0 else remainder
                try:
                    batch = generate_batch(category, config, n)
                    for pair in batch:
                        entry = to_chatml(pair["instruction"], pair["response"], category)
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        cat_done += 1
                        total_done += 1

                    print(f"  call {i+1}/{calls} ✓ [{cat_done}/{target}] — total: {total_done}/{total_target}")
                    time.sleep(5)

                except Exception as e:
                    print(f"  [ERROR] call {i+1}: {e} — skip")
                    time.sleep(15)

    print(f"\nDone! {total_done} pairs saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
