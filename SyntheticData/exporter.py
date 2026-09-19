import os
import logging
from typing import List
import pandas as pd
from .models import QAPair

logger = logging.getLogger(__name__)


class DatasetExporter:
    """
    Handles converting `List[QAPair]` to Pandas DataFrame and exporting
    to various file formats (CSV, Excel, JSON).
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def to_dataframe(pairs: List[QAPair]) -> pd.DataFrame:
        """Convert a list of validated QAPair instances to a Pandas DataFrame."""
        if not pairs:
            return pd.DataFrame(columns=['question', 'answer', 'context', 'chunk_id', 'source_file', 'generated_at'])

        data = [
            {
                'question': pair.question,
                'answer': pair.answer,
                'context': pair.context,
                'chunk_id': pair.chunk_id,
                'source_file': pair.source_file,
                'generated_at': pair.generated_at.isoformat()
            }
            for pair in pairs
        ]
        return pd.DataFrame(data)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        filename_prefix: str = "synthetic_qa_data",
        formats: List[str] = None
    ) -> List[str]:
        """Save DataFrame to specified file formats (csv, excel, json)."""
        formats = formats or ["csv", "excel"]
        saved_paths = []

        for fmt in formats:
            fmt_lower = fmt.lower().strip()
            if fmt_lower == "csv":
                path = os.path.join(self.output_dir, f"{filename_prefix}.csv")
                df.to_csv(path, index=False, encoding='utf-8')
                saved_paths.append(path)
            elif fmt_lower == "excel":
                path = os.path.join(self.output_dir, f"{filename_prefix}.xlsx")
                df.to_excel(path, index=False)
                saved_paths.append(path)
            elif fmt_lower == "json":
                path = os.path.join(self.output_dir, f"{filename_prefix}.json")
                df.to_json(path, orient="records", force_ascii=False, indent=2)
                saved_paths.append(path)

        logger.info(f"Dataset saved to formats {formats} in: {self.output_dir}")
        return saved_paths

    def append_checkpoint_csv(
        self,
        new_pairs: List[QAPair],
        filename: str = "synthetic_qa_data.csv"
    ) -> str:
        """Append new QAPair records atomically to a single CSV checkpoint file."""
        if not new_pairs:
            return ""

        df = self.to_dataframe(new_pairs)
        csv_path = os.path.join(self.output_dir, filename)
        file_exists = os.path.exists(csv_path)

        df.to_csv(
            csv_path,
            mode='a',
            index=False,
            header=not file_exists,
            encoding='utf-8'
        )
        logger.info(f"💾 Checkpoint appended {len(new_pairs)} Q&A pairs to: {csv_path}")
        return csv_path
