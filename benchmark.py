import time
import requests
from bs4 import BeautifulSoup
import mechanicalsoup

TEST_URLS = [
"https://arxiv.org/html/2610.15239",
"https://arxiv.org/html/2609.38187",
"https://arxiv.org/html/2609.34780",
"https://arxiv.org/html/2609.14819",
"https://arxiv.org/html/2609.11990",
"https://arxiv.org/html/2609.09140",
"https://arxiv.org/html/2608.28662",
"https://arxiv.org/html/2608.15424",
"https://arxiv.org/html/2608.06322",
"https://arxiv.org/html/2607.22762"
] 


def benchmark_bs4(urls):
    """Benchmarks Requests + BeautifulSoup4 on static HTML pages."""
    start = time.perf_counter()
    titles = []
    for url in urls:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        if soup.title:
            titles.append(soup.title.text.strip())
    return time.perf_counter() - start, len(titles)


def benchmark_mechanicalsoup(urls):
    """Benchmarks MechanicalSoup StatefulBrowser on the same HTML pages."""
    start = time.perf_counter()
    browser = mechanicalsoup.StatefulBrowser(user_agent="Mozilla/5.0")
    titles = []
    for url in urls:
        browser.open(url, timeout=10)
        page = browser.page
        if page and page.title:
            titles.append(page.title.text.strip())
    return time.perf_counter() - start, len(titles)


def save_benchmark_md(t_bs4, t_ms, count):
    """Saves the benchmark comparison table to benchmark.md."""
    md_content = f"""### Benchmark Results ({count} Pages)

| Metric | Requests + BeautifulSoup4 | MechanicalSoup |
| :--- | :--- | :--- |
| **Execution Time ({count} pages)** | **{t_bs4:.2f} seconds** | **{t_ms:.2f} seconds** |
| **Average Speed per Page** | {t_bs4 / count:.2f}s / page | {t_ms / count:.2f}s / page |
| **Session / State Overhead** | Stateless (minimal memory) | Stateful browser object (tracks history/forms) |
"""
    with open("benchmark.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Done! Benchmark table saved to 'benchmark.md'.")


if __name__ == "__main__":
    print("Running scraping benchmark on 10 pages...")
    t_bs4, count_bs4 = benchmark_bs4(TEST_URLS)
    print(f"Requests + BeautifulSoup4: {t_bs4:.2f}s ({count_bs4} pages)")

    t_ms, count_ms = benchmark_mechanicalsoup(TEST_URLS)
    print(f"MechanicalSoup:            {t_ms:.2f}s ({count_ms} pages)")

    save_benchmark_md(t_bs4, t_ms, len(TEST_URLS))
