# -*- coding: utf-8 -*-
import re
import click
import logging
from pathlib import Path
import pandas as pd

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')

def clean_text(text: str) -> str:
    """basic normalizatn shared by sms and email text"""
    if not isinstance(text,str):
        return""
    text = re.sub(r"<[^>]+>", "", text)  # remove html tags
    text = re.sub(r"http\S+|www\.\S+","URL",text)
    text = re.sub(r"\S+@\S+","EMAIL",text)
    text = re.sub(r"\s+","",text).strip()
    return text

def load_sms(path:Path) -> pd.DataFrame:
    """load sms data"""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: 'label',df.columns[1]: 'text'})
    df['label'] = df['label'].map({'spam':1,'ham':0,1:1,0:0})

    df['source'] = 'sms'
    return df[['text','label','source']]

def load_email(path :Path) -> pd.DataFrame:
    """load email data"""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: 'label',df.columns[1]: 'text'})
    df['label'] = df['label'].map({'spam':1,'ham':0,'not spam':0,1:1,0:0})
    df['source'] = 'email'
    return df[['text','label','source']]


@click.command()
@click.option('--sms-file',default=str(RAW_DIR / 'sms_spam.csv'))
@click.option('--email-file',default=str(RAW_DIR / 'email_spam.csv'))
@click.option('--output-file',default=str(PROCESSED_DIR / 'spam_dataset.csv'))
def main(sms_file,email_file,output_file):
    logger = logging.getLogger(__name__)
    logger.info('making final dataset from raw data')

    frames = []
    if Path(sms_file).exists():
        frames.append(load_sms(Path(sms_file)))
    else:
        logger.warning(f'missing {sms_file} skipping SMS data')

    if Path(email_file).exists():
        frames.append(load_email(Path(email_file)))
    else:
        logger.warning(f'missing {email_file} skipping Email data')

    if not frames:
        raise FileNotFoundError('No raw data found. download datasets and place in data/raw/')

    df = pd.concat(frames,ignore_index = True)
    df['text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['label']).drop_duplicates(subset=['text'])
    df = df[df['text'].str.len() >0]

    Path(output_file).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_file,index=False)
    logger.info(f'wrote {len(df)} rows to {output_file}')
    logger.info(f'Spam ratio:{df['label'].mean():.3f}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
