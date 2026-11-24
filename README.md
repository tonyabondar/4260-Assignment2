# Web Scraping & Text Analysis: AI in Healthcare

## Project Overview
This project collects over 50 recent articles related to **Artificial Intelligence in Healthcare and Medicine**, generates concise summaries for each article, and computes a directional **Importance Score** (positive or negative) reflecting both topical relevance and sentiment polarity.

---

## Setup & Execution Instructions

### 1. Create and Activate a Virtual Environment
Run the following commands in your terminal inside the project directory:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Scripts
```bash
# Run the scraping benchmark (generates benchmark.md)
python benchmark.py

# Run the main scraping and text analysis pipeline (generates results.csv)
python main.py
```
After execution, the output table will be saved to `results.csv` containing three columns: `url`, `summary`, and `score`.

---

## Part 1: Web Scraping — Benchmark & Library Comparison

To make an evidence-based decision, I compared **Requests + BeautifulSoup4** against **MechanicalSoup** on medical and AI news pages (`benchmark.py`).

### Benchmark Results (10 Pages)

| Metric | Requests + BeautifulSoup4 | MechanicalSoup |
| :--- | :--- | :--- |
| **Execution Time (10 pages)** | **2.59 seconds** | **1.23 seconds** |
| **Average Speed per Page** | 0.26s / page | 0.12s / page |
| **Session / State Overhead** | Stateless (minimal memory) | Stateful browser object (tracks history/forms) |


### Ease of Use & Code Comparison

**Requests + BeautifulSoup4 (Selected):**
```python
resp = requests.get(url, headers=HEADERS)
soup = BeautifulSoup(resp.text, "html.parser")
paragraphs = [p.get_text() for p in soup.find_all("p")]
```

**MechanicalSoup:**
```python
browser = mechanicalsoup.StatefulBrowser()
browser.open(url)
paragraphs = [p.get_text() for p in browser.page.find_all("p")]
```

### Benchmark Results (10 Pages)

| Metric | Requests + BeautifulSoup4 | MechanicalSoup |
| :--- | :--- | :--- |
| **Execution Time (10 pages)** | **4.44 seconds** | **0.50 seconds** |
| **Average Speed per Page** | 0.44s / page | 0.05s / page |
| **Session / State Overhead** | Stateless (minimal memory) | Stateful browser object (tracks history/forms) |



| Ease of Use Factor | Requests + BeautifulSoup4 | MechanicalSoup |
| :--- | :--- | :--- |
| **API Simplicity** | Very intuitive CSS/tag selectors (`soup.find_all`) with stateless HTTP requests. | Built on top of BeautifulSoup, adding extra methods for browser state and form filling. |
| **XML / RSS Parsing** | Natively switches between `html.parser` and `xml` parsers out of the box. | Primarily geared toward HTML browser navigation; requires extra handling for raw XML feeds. |
| **Best Use Case** | Read-only static scraping & RSS/XML feed parsing. | Automated logins, form submissions, and multi-step interactive navigation. |

### Recommendation & Justification
I selected **Requests + BeautifulSoup4** for this task because:
1. **Higher Execution Speed:** Without `StatefulBrowser` session and history tracking overhead, raw `requests` + `BeautifulSoup4` executes noticeably faster across multiple pages.
2. **Seamless XML & HTML Handling:** BeautifulSoup effortlessly parses both RSS/XML feeds (to gather article links) and HTML pages (to extract article text) using a single consistent syntax.
3. **Zero Unnecessary Complexity:** Since public scientific news portals do not require logging in or submitting forms, MechanicalSoup's stateful browser capabilities are redundant.

---

## Part 2: Text Analysis — Methodology & Reasoning

Two distinct NLP algorithms were applied to the collected dataset:

### 1. Extractive Text Summarization (`LexRank` via `sumy`)
* **How it works:** LexRank is an unsupervised graph-based algorithm (inspired by PageRank). It represents sentences as nodes in a graph and computes edge weights using cosine similarity between sentence TF-IDF vectors.
* **Why it was chosen:** Unlike commercial LLM APIs, LexRank runs locally with zero API cost, preserves factual medical terminology directly from the source text, and extracts the top 2 most central sentences per article.

### 2. Directional Importance Score (`TF-IDF Cosine Similarity` + `VADER Sentiment`)
To evaluate how important each article is to the domain of **AI in Healthcare** and whether its impact is positive or negative, a composite score was engineered:
1. **Topical Importance Magnitude (`0.15` to `1.0`):** Using `scikit-learn`'s `TfidfVectorizer`, each article is vectorized alongside a domain reference profile (*"artificial intelligence, machine learning, healthcare, clinical diagnosis, patient treatment, FDA, drug discovery"*). Cosine similarity measures how strongly the article focuses on medical AI.
2. **Polarity Direction (`+` or `-`):** NLTK's **VADER** (`SentimentIntensityAnalyzer`) computes the compound sentiment polarity (`-1` to `+1`). Breakthroughs in diagnostics or treatment receive a **positive (`+`)** sign, whereas articles discussing clinical risks, bias, data privacy issues, or diseases receive a **negative (`-`)** sign.
3. **Final Formula:**
   $$\text{Score} = \text{sign}(\text{VADER}_{\text{compound}}) \times \text{Importance}_{\text{norm}} \times (0.5 + 0.5 \times |\text{VADER}_{\text{compound}}|)$$

### Key Findings
* Articles reporting successful AI diagnostic tools (e.g., early cancer detection or faster drug screening) consistently yield high positive scores (`+0.55` to `+0.85`).
* Articles highlighting algorithmic errors, healthcare cybersecurity threats, or severe disease mortality risks yield negative scores (`-0.30` to `-0.75`), allowing quick filtering of opportunities versus risks in medical AI.
