
import os
import wikipediaapi

DATA_DIR = "./data"

def ensure_data_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def fetch_and_save_pages(titles: list[str], language: str = "en") -> None:
    ensure_data_dir(DATA_DIR)
    wiki = wikipediaapi.Wikipedia(language=language, user_agent='LangChain-Project/1.0.0 (https://github.com/your-username/your-project)')

    for title in titles:
        page = wiki.page(title)
        if not page.exists():
            print(f"[SKIP] Page not found: {title}")
            continue

        safe_name = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(DATA_DIR, f"{safe_name}.txt")

        print(f"[WRITE] {title} -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page.title + "\n\n")
            f.write(page.summary + "\n\n")
            f.write(page.text)

if __name__ == "__main__":

    topics = [
        "Software engineering",
        "Software testing",
        "Requirements engineering",
        "Software design"
    ]
    fetch_and_save_pages(topics)