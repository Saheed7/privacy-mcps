"""
Data Preprocessing Pipeline (DPMS Microservice Logic).

Implements data loading, cleaning, scaling, SMOTE, partitioning
for CIC-IoMT2024 and Edge-IIoTset datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class DataPipeline:
    """End-to-end data preprocessing pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        ds_config = self.config["dataset"][dataset_name]
        path = ds_config["path"]
        logger.info(f"Loading {ds_config['name']} from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} samples, {len(df.columns)} columns")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove NaN, Inf, duplicate, and constant columns."""
        initial_shape = df.shape
        # Drop non-numeric identifiers and timestamps
        cols_to_drop = [c for c in df.columns if df[c].dtype == "object" and c != self.config["dataset"].get("label_column", "label")]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        # Replace infinities and drop NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        # Remove constant columns
        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        df.drop(columns=constant_cols, inplace=True)
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        logger.info(f"Cleaned: {initial_shape} -> {df.shape}")
        return df

    def prepare_features_labels(
        self, df: pd.DataFrame, label_column: str, task: str = "binary"
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Extract features and encode labels."""
        y_raw = df[label_column].values
        X = df.drop(columns=[label_column]).values.astype(np.float32)

        if task == "binary":
            # Encode: benign/normal=0, all attacks=1
            y_raw_lower = pd.Series(y_raw).str.lower().str.strip()
            benign_labels = ["benign", "normal", "0", "legitimate"]
            y = np.where(y_raw_lower.isin(benign_labels), 0, 1).astype(np.int32)
            n_classes = 2
        else:
            y = self.label_encoder.fit_transform(y_raw)
            n_classes = len(self.label_encoder.classes_)
            y = to_categorical(y, num_classes=n_classes).astype(np.float32)

        logger.info(f"Features: {X.shape}, Classes: {n_classes}, Task: {task}")
        return X, y, n_classes

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Min-Max scaling (fitted on train only)."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)

    def apply_smote(self, X: np.ndarray, y: np.ndarray, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for class imbalance."""
        if not self.config["preprocessing"]["smote"]:
            return X, y
        logger.info("Applying SMOTE oversampling...")
        if task == "multiclass":
            y_flat = np.argmax(y, axis=1)
        else:
            y_flat = y
        smote = SMOTE(random_state=self.config["preprocessing"]["smote_random_state"])
        X_res, y_res = smote.fit_resample(X, y_flat)
        if task == "multiclass":
            n_classes = y.shape[1]
            y_res = to_categorical(y_res, num_classes=n_classes).astype(np.float32)
        logger.info(f"After SMOTE: {X_res.shape[0]} samples")
        return X_res.astype(np.float32), y_res

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train/test split."""
        prep = self.config["preprocessing"]
        if y.ndim > 1:
            stratify_y = np.argmax(y, axis=1)
        else:
            stratify_y = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=prep["test_size"],
            random_state=prep["random_state"],
            stratify=stratify_y,
        )
        logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def partition_data(
        self, X: np.ndarray, y: np.ndarray, n_partitions: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data into M partitions for distributed training."""
        X_parts = np.array_split(X, n_partitions)
        y_parts = np.array_split(y, n_partitions)
        partitions = list(zip(X_parts, y_parts))
        sizes = [len(xp) for xp in X_parts]
        logger.info(f"Partitioned into {n_partitions} segments: {sizes}")
        return partitions

    def full_pipeline(
        self, dataset_name: str, task: str = "binary"
    ) -> Dict:
        """Run the complete preprocessing pipeline."""
        ds_cfg = self.config["dataset"][dataset_name]

        # Load and clean
        df = self.load_dataset(dataset_name)
        df = self.clean_data(df)

        # Features and labels
        X, y, n_classes = self.prepare_features_labels(df, ds_cfg["label_column"], task)

        # Split
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Scale
        X_train, X_test = self.scale_features(X_train, X_test)

        # SMOTE (on training set only)
        X_train, y_train = self.apply_smote(X_train, y_train, task)

        # Shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "n_features": X_train.shape[1],
            "n_classes": n_classes,
            "dataset_name": ds_cfg["name"],
            "task": task,
        }
