import pandas as pd
from typing import List, Tuple, Dict
import logging
import zipfile
import os
from src.mcd.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

def resolve_dataset_path(path: str) -> Tuple[str, str]:
    """Resolve dataset path: if ZIP, extract and select CSV; if CSV, return as is."""
    if path.endswith('.zip'):
        if not os.path.exists(path):
            raise ValueError(f"Archive not found: {path}")
        extract_dir = os.path.join(os.path.dirname(path), 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        priority_files = [
            'dataset-tickets-multi-lang-4-20k.csv',
            'aa_dataset-tickets-multi-lang-5-2-50-version.csv'
        ]
        
        csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in extracted archive {path}")
        
        selected_csv = None
        for priority in priority_files:
            if priority in csv_files:
                selected_csv = priority
                break
        if not selected_csv:
            selected_csv = csv_files[0]
        
        csv_path = os.path.join(extract_dir, selected_csv)
        logger.info(f"Selected CSV from archive: {csv_path}")
        return csv_path, selected_csv
    else:
        return path, os.path.basename(path)

def load_labeled_tickets_csv(path: str, label_column: str) -> Tuple[List[str], List[str], Dict[str, int], Dict[int, str]]:
    """Load labeled tickets from CSV."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read CSV {path}: {e}")
        raise

    required_cols = ['subject', 'body', label_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=required_cols)

    texts = []
    for _, row in df.iterrows():
        text = f"{row['subject']}\n\n{row['body']}"
        text = preprocess_text(text)  # Apply preprocessing
        texts.append(text)

    labels = df[label_column].tolist()

    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    logger.info(f"Loaded {len(texts)} samples with {len(unique_labels)} unique labels")
    return texts, labels, label_to_index, index_to_label