import re
import logging

import click
import pandas as pd

from config import SMS_RAW_PATH, EMAIL_RAW_PATH, PROCESSED_DATA_PATH


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\.\S+", "URL", text)
    text = re.sub(r"\S+@\S+", "EMAIL", text)
    return text


def load_sms(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df["label"] = df["label"].map({"spam": 1, "ham": 0, 1: 1, 0: 0})
    df["source"] = "sms"
    return df[["text", "label", "source"]]


def load_email(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df["label"] = df["label"].map({"spam": 1, "ham": 0, "not spam": 0, 1: 1, 0: 0})
    df["source"] = "email"
    return df[["text", "label", "source"]]


@click.command()
@click.option("--sms-file", default=str(SMS_RAW_PATH))
@click.option("--email-file", default=str(EMAIL_RAW_PATH))
@click.option("--output-file", default=str(PROCESSED_DATA_PATH))
def main(sms_file, email_file, output_file):
    from pathlib import Path
    logger = logging.getLogger(__name__)
    logger.info("Loading raw datasets")

    frames = []
    if Path(sms_file).exists():
        frames.append(load_sms(Path(sms_file)))
    else:
        logger.warning(f"Missing {sms_file}, skipping SMS data")

    if Path(email_file).exists():
        frames.append(load_email(Path(email_file)))
    else:
        logger.warning(f"Missing {email_file}, skipping email data")

    if not frames:
        raise FileNotFoundError("No raw data found. Place files in data/raw/")

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["label"]).drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() > 0]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Wrote {len(df)} rows to {output_file}")
    logger.info(f"Spam ratio: {df['label'].mean():.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
