import requests
from bs4 import BeautifulSoup
import random

# 你可以修改这个列表，换成你更喜欢的主题
WIKI_PAGES = [
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/History_of_science",
    "https://en.wikipedia.org/wiki/Universe",
    "https://en.wikipedia.org/wiki/Human_brain",
    "https://en.wikipedia.org/wiki/World_War_II",
]

def fetch_wiki_text(url):
    """抓取维基百科页面文本"""
    print(f"Fetching: {url}")
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")

    paragraphs = soup.find_all("p")
    texts = []
    for p in paragraphs:
        text = p.get_text().strip()
        if len(text) > 100:  # 避免太短无意义段落
            texts.append(text)

    return texts


def generate_file(filename, approx_size_bytes):
    print(f"\nGenerating {filename} (approx {approx_size_bytes/1024/1024:.2f} MB)")

    # 1. 抓取所有维基文本
    all_texts = []
    for url in WIKI_PAGES:
        try:
            all_texts.extend(fetch_wiki_text(url))
        except:
            continue

    if not all_texts:
        raise Exception("No text fetched from Wikipedia!")

    # 2. 持续写入直到达到大致容量
    with open(filename, "w", encoding="utf-8") as f:
        current_size = 0
        while current_size < approx_size_bytes:
            paragraph = random.choice(all_texts)
            f.write(paragraph + "\n\n")  # 自然段落
            current_size += len(paragraph.encode("utf-8"))

    print(f"{filename} created (around {current_size/1024/1024:.2f} MB)")


# 生成三个文件（不追求精确，只保证差不多）
generate_file("10KB.txt", 10 * 1024)
generate_file("1MB.txt", 1 * 1024 * 1024)
generate_file("100MB.txt", 100 * 1024 * 1024)
