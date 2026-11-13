"""
download_data.py
================
Data loading and preprocessing pipeline.
Uses esg_engine.py as the single source of truth for ESG scoring.
"""

import pandas as pd
import numpy as np
import os


class Data:
    def __init__(self):
        pass

    def _create_empty_data(self):
        """Returns a minimal empty dataset to avoid crashes."""
        return {
            "conn": pd.DataFrame(columns=["company"]),
            "data": pd.DataFrame(columns=["DATE", "Organization", "Tone", "ESG_Score", "SourceCommonName"]),
            "embed": pd.DataFrame(columns=["Organization"]),
            "overall_score": pd.DataFrame(),
            "E_score": pd.DataFrame(),
            "S_score": pd.DataFrame(),
            "G_score": pd.DataFrame(),
            "ESG": pd.DataFrame(),
        }

    def paths(self, data_path, config=None):
        """Dynamic path mapping."""
        cfg = config or {
            "conn": "connections.csv",
            "data": "data_as_csv.csv",
            "embed": "pca_embeddings.csv",
            "avg": "average_esg_scores.csv",
            "daily": "overall_daily_esg_scores.csv",
            "e": "daily_E_score.csv",
            "s": "daily_S_score.csv",
            "g": "daily_G_score.csv",
        }
        self.connections = os.path.join(data_path, cfg["conn"])
        self.data = os.path.join(data_path, cfg["data"])
        self.embeddings = os.path.join(data_path, cfg["embed"])
        esg_path = os.path.join(data_path, "ESG")
        self.avg_esg = os.path.join(esg_path, cfg["avg"])
        self.daily_esg = os.path.join(esg_path, cfg["daily"])
        self.e_score = os.path.join(esg_path, cfg["e"])
        self.s_score = os.path.join(esg_path, cfg["s"])
        self.g_score = os.path.join(esg_path, cfg["g"])

    def read(self, start_day="jan1", end_day="jan5"):
        """Reads data using Pure Pandas."""
        # Handle both standard patterns and single-name datasets
        if start_day == end_day:
            dir_name = start_day
        else:
            dir_name = f"{start_day}_to_{end_day}"

        if not os.path.exists("Data") or dir_name not in os.listdir("Data"):
            raise NameError(f"There isn't data for {dir_name}")

        data_path = os.path.join(".", "Data", dir_name)
        self.paths(data_path)

        if not os.path.exists(self.data):
            print(f"Warning: Data file {self.data} not found.")
            return self._create_empty_data()

        # 1. Load connections
        try:
            pdf_conn = pd.read_csv(self.connections)
            if "company" not in pdf_conn.columns:
                cols = ["company"] + [f"n{i}_{t}" for i in range(25) for t in ["rec", "conf"]]
                pdf_conn = pd.read_csv(self.connections, header=None, names=cols)
        except Exception as e:
            print(f"[WARNING] Failed to load connections: {e}")
            pdf_conn = pd.DataFrame(columns=["company"])

        # 2. Ingest main news data
        pdf_data = pd.read_csv(self.data, low_memory=False)

        # Force numeric on key columns
        numeric_cols = ["Tone", "AvgTone", "NumArticles", "NumMentions", "WordCount",
                        "ActivityDensity", "SelfDensity"]
        for col in numeric_cols:
            if col in pdf_data.columns:
                pdf_data[col] = pd.to_numeric(pdf_data[col], errors="coerce")

        # Ensure AvgTone / NumArticles / NumMentions exist (fallbacks)
        if "AvgTone" not in pdf_data.columns and "Tone" in pdf_data.columns:
            pdf_data["AvgTone"] = pdf_data["Tone"]
        if "NumArticles" not in pdf_data.columns:
            pdf_data["NumArticles"] = pdf_data.get("WordCount", pd.Series(1, index=pdf_data.index))
        if "NumMentions" not in pdf_data.columns:
            pdf_data["NumMentions"] = pdf_data.get("ActivityDensity", pd.Series(1, index=pdf_data.index))

        # Drop rows with missing Tone
        if "Tone" in pdf_data.columns:
            pdf_data = pdf_data.dropna(subset=["Tone"])

        # Remove duplicates
        if "URL" in pdf_data.columns:
            before = len(pdf_data)
            pdf_data = pdf_data.drop_duplicates(subset=["DATE", "Organization", "URL"], keep="first")
            removed = before - len(pdf_data)
            if removed > 0:
                print(f"[INFO] Removed {removed:,} duplicate rows")

        # 3. Load embeddings
        try:
            pdf_embed = pd.read_csv(self.embeddings)
        except Exception:
            if "Organization" in pdf_data.columns:
                unique_orgs = pdf_data["Organization"].dropna().unique()
            else:
                unique_orgs = []
            pdf_embed = pd.DataFrame({"Organization": unique_orgs})
            for i in range(10):
                pdf_embed[f"c{i}"] = 0.0

        # 4. Date standardization
        if "DATE" not in pdf_data.columns and "date" in pdf_data.columns:
            pdf_data = pdf_data.rename(columns={"date": "DATE"})

        if "DATE" in pdf_data.columns:
            pdf_data["DATE"] = pd.to_datetime(
                pdf_data["DATE"], dayfirst=True, format="mixed", errors="coerce"
            )

        pdf_data = pdf_data.dropna(subset=["DATE"])
        pdf_data["DATE"] = pdf_data["DATE"].dt.date

        # 5. Build processed data dict
        processed_data = {
            "conn": pdf_conn,
            "data": pdf_data,
            "embed": pdf_embed,
        }

        # 6. Process ESG Pillar tables
        esg_tables = ["overall_score", "E_score", "S_score", "G_score", "ESG"]
        esg_paths = [self.daily_esg, self.e_score, self.s_score, self.g_score, self.avg_esg]

        for name, path in zip(esg_tables, esg_paths):
            if not os.path.exists(path):
                print(f"[WARNING] Missing ESG file: {path}")
                processed_data[name] = pd.DataFrame()
                continue

            pdf_esg = pd.read_csv(path)

            # Multiply numeric columns by 10000 (legacy format)
            num_cols = pdf_esg.select_dtypes(include=[np.number]).columns
            pdf_esg[num_cols] = pdf_esg[num_cols] * 10000

            # Standardize date column
            if "date" in pdf_esg.columns:
                pdf_esg["date"] = pd.to_datetime(
                    pdf_esg["date"], dayfirst=True, format="mixed", errors="coerce"
                )
                if name != "ESG":
                    pdf_esg = pdf_esg.set_index("date")
            elif "DATE" in pdf_esg.columns:
                pdf_esg["DATE"] = pd.to_datetime(
                    pdf_esg["DATE"], dayfirst=True, format="mixed", errors="coerce"
                )
                if name != "ESG":
                    pdf_esg = pdf_esg.set_index("DATE")

            processed_data[name] = pdf_esg

        return processed_data
