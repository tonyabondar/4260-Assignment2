import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Download required NLTK corpora silently
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("vader_lexicon", quiet=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

TOPIC_REFERENCE = (
    "artificial intelligence machine learning deep learning healthcare medicine "
    "clinical diagnosis patient treatment hospital medical imaging cancer therapy"
)


def scrape_arxiv_medical_ai(target_count=60):
    """
    Scrapes HTML search results from arXiv for 'artificial intelligence healthcare medicine'
    using Requests + BeautifulSoup4.
    """
    articles = []
    base_url = "https://arxiv.org/search/"
    query_params = {
        "query": "artificial intelligence healthcare medicine",
        "searchtype": "all",
        "source": "header",
        "size": "100"
    }

    try:
        print(f"Sending HTTP GET request to {base_url}...")
        resp = requests.get(base_url, params=query_params, headers=HEADERS, timeout=20)
        print(f"HTTP status code: {resp.status_code}")
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("li", class_="arxiv-result")
        print(f"Found article cards on page: {len(results)}")

        for item in results:
            if len(articles) >= target_count:
                break

            # Extract article URL
            link_tag = item.find("p", class_="list-title")
            a_tag = link_tag.find("a") if link_tag else None
            url = a_tag["href"].strip() if a_tag and a_tag.has_attr("href") else ""
            # Convert abstract URL to HTML URL
            url = url.replace("/abs/", "/html/")

            # Extract article title
            title_tag = item.find("p", class_="title")
            title = title_tag.get_text(" ", strip=True) if title_tag else ""

            # Extract full abstract text
            abstract_tag = item.find("span", class_="abstract-full")
            if not abstract_tag:
                abstract_tag = item.find("p", class_="abstract")
            abstract = abstract_tag.get_text(" ", strip=True) if abstract_tag else ""
            # Remove trailing 'Less' toggle text from arXiv abstracts
            abstract = re.sub(r"\s*△ Less\s*$", "", abstract).strip()

            full_text = f"{title}. {abstract}"
            if url and len(full_text) > 150:
                articles.append({
                    "url": url,
                    "title": title,
                    "text": full_text
                })

    except Exception as e:
        print(f"CRITICAL ERROR during scraping: {type(e).__name__}: {e}")

    return articles


def summarize_text(text, sentences_count=2):
    """Algorithm 1: Extractive text summarization using LexRank (sumy library)."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count)
        summary = " ".join(str(sentence) for sentence in summary_sentences).strip()
        return summary if summary else text[:250] + "..."
    except Exception:
        return text[:250] + "..."


def compute_importance_scores(texts):
    """
    Algorithm 2: Directional Importance Score calculation.
    1. Topical importance (0.15..1.0) via TF-IDF + Cosine Similarity against the domain profile.
    2. Polarity direction (+ or -) via VADER SentimentIntensityAnalyzer.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1500)
    corpus = [TOPIC_REFERENCE] + texts
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    max_sim = similarities.max() if similarities.max() > 0 else 1.0
    normalized_importance = 0.15 + 0.85 * (similarities / max_sim)

    sia = SentimentIntensityAnalyzer()
    final_scores = []

    for text, importance in zip(texts, normalized_importance):
        sentiment = sia.polarity_scores(text)["compound"]
        direction = 1 if sentiment >= 0 else -1
        weighted_score = direction * round(float(importance * (0.5 + 0.5 * abs(sentiment))), 4)
        final_scores.append(weighted_score)

    return final_scores


def main():
    print("1. Scraping AI in Healthcare articles from arXiv...")
    articles = scrape_arxiv_medical_ai(target_count=60)
    print(f"Successfully collected articles: {len(articles)}")

    if not articles:
        print("No articles were collected. Please inspect the error message above.")
        return

    raw_texts = [a["text"] for a in articles]

    print("2. Generating extractive summaries (LexRank)...")
    summaries = [summarize_text(t, sentences_count=2) for t in raw_texts]

    print("3. Computing directional Importance Scores (TF-IDF + VADER)...")
    scores = compute_importance_scores(raw_texts)

    df = pd.DataFrame({
        "url": [a["url"] for a in articles],
        "summary": summaries,
        "score": scores
    })

    output_file = "results.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nDone! Saved {len(df)} articles to '{output_file}'.")
    print("\nFirst 5 rows of the output table:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
