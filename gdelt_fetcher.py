import pandas as pd
import requests
import zipfile
import io
import os
import sys
from datetime import datetime, timedelta
import argparse
import hashlib
import json

class GDELTFetcher:
    LAST_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    MASTER_LIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
    
    # ESG Keyword/Theme mapping
    E_THEMES = ['ENV_', 'CLIMATECHANGE', 'POLLUTION', 'RENEWABLE', 'NATURAL_RESOURCES', 'WATER_SCARCITY', 'GREEN_ENERGY']
    S_THEMES = ['SOC_', 'HUMAN_RIGHTS', 'LABOR_ISSUES', 'DIVERSITY', 'INCLUSION', 'PUBLIC_HEALTH', 'DISCRIMINATION']
    G_THEMES = ['GOV_', 'CORRUPTION', 'WB_REGULATORY_REFORM', 'FRAUD', 'TRANSPARENCY', 'CORPORATE_GOVERNANCE', 'TAX_EVASION']

    def __init__(self, output_dir="Data/GDELT_Live"):
        self.output_dir = output_dir
        self.TARGET_COMPANIES = self._load_target_companies()

    def _load_target_companies(self):
        """Loads company keys from profile JSON for filtering."""
        try:
            with open("company_profiles.json", "r") as f:
                profiles = json.load(f)
                # Store as uppercase for fast case-insensitive matching in GKG loop
                return set([k.upper() for k in profiles.keys()])
        except Exception as e:
            print(f"Warning: Could not load company_profiles.json: {e}")
            return set()

    def _ensure_dirs(self):
        """Creates the output directories only when needed."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        esg_dir = os.path.join(self.output_dir, "ESG")
        if not os.path.exists(esg_dir):
            os.makedirs(esg_dir)

    def get_latest_gkg_url(self):
        """Fetches the URL of the most recent GKG 2.0 file."""
        try:
            response = requests.get(self.LAST_UPDATE_URL, timeout=10)
            if response.status_code == 200:
                lines = response.text.split('\n')
                for line in lines:
                    if 'gkg.csv.zip' in line:
                        return line.split(' ')[2]
        except Exception as e:
            print(f"Error fetching GDELT last update: {e}")
        return None

    def get_historical_urls(self, start_date, end_date):
        """Finds one GKG file per day in the range [start_date, end_date]."""
        print(f"Fetching master file list from GDELT...")
        response = requests.get(self.MASTER_LIST_URL, timeout=30)
        lines = response.text.split('\n')
        
        urls = {}
        for line in lines:
            if 'gkg.csv.zip' not in line: continue
            parts = line.split(' ')
            if len(parts) < 3: continue
            url = parts[2]
            timestamp_str = url.split('/')[-1].split('.')[0]
            try:
                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                if start_date <= dt <= end_date:
                    day_key = dt.strftime("%Y%m%d")
                    if day_key not in urls:
                        urls[day_key] = url
            except:
                continue
        
        return sorted(list(urls.values()))

    def download_and_load(self, url):
        """Downloads the zip, extracts and loads into a Pandas DataFrame."""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # Column indices: 1=DATE, 3=Source, 4=URL, 8=V2Themes, 10=V2Locations, 12=V2Persons, 14=V2Orgs, 15=Tone
                    df = pd.read_csv(f, sep='\t', header=None, low_memory=False, usecols=[1, 3, 4, 8, 10, 12, 14, 15])
                    df.columns = ['DATE_RAW', 'SourceCommonName', 'URL', 'V2Themes', 'V2Locations', 'V2Persons', 'V2Organizations', 'V2Tone']
                    return df
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None

    def process_data(self, df):
        """Processes raw GKG data to match the project's schema and ML-ready requirements."""
        if df is None or df.empty: return pd.DataFrame()
        
        # Deduplication by URL
        df = df.drop_duplicates(subset=['URL'])
        
        processed_rows = []
        for idx, row in df.iterrows():
            try:
                date_raw = str(row['DATE_RAW'])
                if len(date_raw) < 14: continue
                date_obj = datetime.strptime(date_raw[:14], "%Y%m%d%H%M%S")
                date_str = date_obj  # store as datetime object directly
                
                themes = str(row['V2Themes'])
                orgs = str(row['V2Organizations'])
                persons = str(row['V2Persons'])
                locs = str(row['V2Locations'])
                tone_str = str(row['V2Tone'])
                
                tone_data = tone_str.split(',')
                if len(tone_data) < 6: continue
                
                # ESG mapping
                is_e = any(t in themes for t in self.E_THEMES)
                is_s = any(t in themes for t in self.S_THEMES)
                is_g = any(t in themes for t in self.G_THEMES)
                
                try:
                    tone_val = float(tone_data[0])
                except:
                    continue
                
                # Weighted Intelligence Scoring
                # Formula: 0.4*E + 0.3*S + 0.3*G
                # If an article mentions multiple pillars, they contribute to the score.
                # If none, it's a general news item (score = 0 for ESG purposes)
                e_weight = 0.4 if is_e else 0.0
                s_weight = 0.3 if is_s else 0.0
                g_weight = 0.3 if is_g else 0.0
                
                total_weight = e_weight + s_weight + g_weight
                # Normalize weight if multiple pillars present, otherwise use raw tone for the matched pillar
                max_weight = 0.4 + 0.3 + 0.3
                weighted_score = tone_val * (total_weight / max_weight) if total_weight > 0 else 0.0
                
                # Filter: Only keep organizations we are tracking and only news with ESG themes
                org_list = [o.split(',')[0].strip().upper() for o in orgs.split(';') if o]
                org_list = [o for o in org_list if o in self.TARGET_COMPANIES]
                
                if not org_list or weighted_score == 0:
                    continue
                    
                person_list = [p.split(',')[0].strip().title() for p in persons.split(';') if p]
                loc_list = [l.split('#')[1] for l in locs.split(';') if '#' in l] # Extract city/state name
                
                primary_person = person_list[0] if person_list else "UNKNOWN"
                primary_loc = loc_list[0] if loc_list else "GLOBAL"
                
                for org in org_list:
                    processed_rows.append({
                        "DATE": date_str,
                        "Organization": org,
                        "SourceCommonName": row['SourceCommonName'],
                        "URL": row['URL'],
                        "E": 1 if is_e else 0,
                        "S": 1 if is_s else 0,
                        "G": 1 if is_g else 0,
                        "Person": primary_person,
                        "Location": primary_loc,
                        "Tone": tone_val,
                        "AvgTone": tone_val,
                        "NumArticles": 1,
                        "NumMentions": 1,
                        "ESG_Score": weighted_score,
                        "PositiveTone": float(tone_data[1]),
                        "NegativeTone": float(tone_data[2]),
                        "Polarity": float(tone_data[3]),
                        "ActivityDensity": float(tone_data[4]),
                        "SelfDensity": float(tone_data[5]),
                        "WordCount": int(tone_data[6]) if len(tone_data) > 6 else 0
                    })
            except: continue
        
        pdf = pd.DataFrame(processed_rows)
        if not pdf.empty:
            # Fill NaNs for safety
            # Fill only categorical columns
            for col in ["Person", "Location", "SourceCommonName"]:
                if col in pdf.columns:
                    pdf[col] = pdf[col].fillna("UNKNOWN")
                    # ✅ ADD THIS BLOCK HERE
            numeric_cols = ["Tone", "ESG_Score", "PositiveTone", "NegativeTone", "Polarity", "ActivityDensity", "SelfDensity"]

            for col in numeric_cols:
                if col in pdf.columns:
                    pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

            pdf = pdf.dropna(subset=["Tone"])
        return pdf

    def generate_aggregates(self, df):
        """Generates the ESG aggregate files in PIVOTED format matching dec30_to_jan12 schema."""
        if df.empty: return {}
        import numpy as np

        # Error Fix: Use flexible date parsing to match our new ISO format
        df['dt_obj'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
        df['date'] = df['dt_obj'].dt.date
        df = df.dropna(subset=["dt_obj"])
        
        # Benchmarking: Industry tone (average of all news)
        industry_tone = df.groupby('date')['Tone'].mean().reset_index().rename(columns={'Tone': 'industry_tone'})

        # Helper: build a pivoted daily score table with _diff suffix
        def _pivoted_daily(subset, value_col='Tone'):
            if subset.empty:
                return pd.DataFrame(columns=['date', 'industry_tone'])
            daily = subset.groupby(['date', 'Organization'])[value_col].mean().reset_index()
            pivoted = daily.pivot(index='date', columns='Organization', values=value_col)
            pivoted.columns = [f"{c.replace(' ', '_')}_diff" for c in pivoted.columns]
            pivoted = pivoted.reset_index()
            pivoted = pivoted.merge(industry_tone, on='date', how='left')
            cols = ['date', 'industry_tone'] + [c for c in pivoted.columns if c not in ('date', 'industry_tone')]
            return pivoted[cols]

        # 1. Average ESG scores (PIVOTED: companies as columns, E/S/G as rows)
        import itertools as _it
        orgs = sorted(df['Organization'].unique())
        avg_data = {}
        for org in orgs:
            org_df = df[df['Organization'] == org]
            e_tone = org_df.loc[org_df['E'].astype(bool), 'Tone'].mean() if org_df['E'].astype(bool).any() else np.nan
            s_tone = org_df.loc[org_df['S'].astype(bool), 'Tone'].mean() if org_df['S'].astype(bool).any() else np.nan
            g_tone = org_df.loc[org_df['G'].astype(bool), 'Tone'].mean() if org_df['G'].astype(bool).any() else np.nan
            avg_data[org] = [e_tone, s_tone, g_tone]
        avg_esg = pd.DataFrame(avg_data, index=['E', 'S', 'G'])
        avg_esg = avg_esg.reset_index().rename(columns={'index': ''})
        
        # 2. Daily overall score
        score_col = 'ESG_Score' if 'ESG_Score' in df.columns else 'Tone'
        daily_overall = _pivoted_daily(df, value_col=score_col)
        
        # 3. Daily Pillar scores
        daily_e = _pivoted_daily(df[df['E'].astype(bool)])
        daily_s = _pivoted_daily(df[df['S'].astype(bool)])
        daily_g = _pivoted_daily(df[df['G'].astype(bool)])
        
        return {
            "average_esg_scores": avg_esg,
            "overall_daily_esg_scores": daily_overall,
            "daily_E_score": daily_e,
            "daily_S_score": daily_s,
            "daily_G_score": daily_g
        }

    def _save_connections_and_embeddings(self, full_df):
        """Generates co-occurrence connections and PCA embeddings."""
        import itertools
        
        # --- Connections: co-occurrence graph ---
        url_orgs = full_df.groupby("URL")["Organization"].apply(list).reset_index()
        url_orgs = url_orgs[url_orgs["Organization"].apply(len) > 1]
        
        edge_weights = {}
        for _, row in url_orgs.iterrows():
            orgs = list(set(row["Organization"]))
            for a, b in itertools.combinations(sorted(orgs), 2):
                key = (a, b)
                edge_weights[key] = edge_weights.get(key, 0) + 1
        
        neighbor_map = {}
        for (a, b), w in edge_weights.items():
            neighbor_map.setdefault(a, []).append((b, w))
            neighbor_map.setdefault(b, []).append((a, w))
        for org in neighbor_map:
            neighbor_map[org].sort(key=lambda x: -x[1])
        
        all_orgs = sorted(full_df["Organization"].unique())
        rows = []
        for org in all_orgs:
            row = {"company": org}
            neighbors = neighbor_map.get(org, [])
            for i in range(25):
                if i < len(neighbors):
                    row[f"n{i}_rec"] = neighbors[i][0]
                    row[f"n{i}_conf"] = neighbors[i][1]
                else:
                    row[f"n{i}_rec"] = ""
                    row[f"n{i}_conf"] = 0
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(self.output_dir, "connections.csv"), index=False)
        
        # --- PCA Embeddings ---
        feature_cols = ["Tone", "PositiveTone", "NegativeTone", "Polarity", "ActivityDensity", "SelfDensity"]
        safe_cols = [c for c in feature_cols if c in full_df.columns]
        
        if len(safe_cols) >= 2:
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                import numpy as np
                org_features = full_df.groupby("Organization")[safe_cols].mean().fillna(0)
                for col in ["E", "S", "G"]:
                    if col in full_df.columns:
                        org_features[f"{col}_ratio"] = full_df.groupby("Organization")[col].mean()
                X = StandardScaler().fit_transform(org_features.values)
                n_comp = min(10, X.shape[1], X.shape[0])
                components = PCA(n_components=n_comp).fit_transform(X)
                embed_df = pd.DataFrame({"Organization": org_features.index})
                for i in range(n_comp): embed_df[f"c{i}"] = components[:, i]
                for i in range(n_comp, 10): embed_df[f"c{i}"] = 0.0
                embed_df.to_csv(os.path.join(self.output_dir, "pca_embeddings.csv"), index=False)
                return
            except ImportError:
                pass
        
        # Fallback: zero embeddings
        unique_orgs = full_df['Organization'].unique()
        embed_df = pd.DataFrame({'Organization': unique_orgs})
        for i in range(10): embed_df[f'c{i}'] = 0.0
        embed_df.to_csv(os.path.join(self.output_dir, "pca_embeddings.csv"), index=False)

    def fetch_historical(self, start_date_str, end_date_str):
        """Fetches and processes data for a date range."""
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d") + timedelta(days=1)
        
        urls = self.get_historical_urls(start_date, end_date)
        if not urls:
            print("No URLs found in this range.")
            return False
            
        self._ensure_dirs()
        print(f"Found {len(urls)} daily files to process.")
        
        all_dfs = []
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Processing {url}...")
            raw_pdf = self.download_and_load(url)
            processed_pdf = self.process_data(raw_pdf)
            if not processed_pdf.empty:
                all_dfs.append(processed_pdf)
            
            if len(all_dfs) % 20 == 0:
                print("Checkpoint: Partial save...")
                full_df = pd.concat(all_dfs)
                full_df.to_csv(os.path.join(self.output_dir, "data_as_csv.csv"), index=False)
        
        if not all_dfs: return False
            
        full_df = pd.concat(all_dfs)
        full_df.to_csv(os.path.join(self.output_dir, "data_as_csv.csv"), index=False)
        
        aggregates = self.generate_aggregates(full_df)
        for name, data in aggregates.items():
            if data is not None and not data.empty:
                data.to_csv(os.path.join(self.output_dir, "ESG", f"{name}.csv"), index=False)
            else:
                pd.DataFrame(columns=['date']).to_csv(os.path.join(self.output_dir, "ESG", f"{name}.csv"), index=False)
        
        self._save_connections_and_embeddings(full_df)
        print(f"Data saved to {self.output_dir}")
        return True

    def fetch_latest(self):
        """Fetch only the single most recent GKG file."""
        url = self.get_latest_gkg_url()
        if not url: return False
        raw_df = self.download_and_load(url)
        processed_df = self.process_data(raw_df)
        if processed_df.empty: return False
        
        self._ensure_dirs()
        processed_df.to_csv(os.path.join(self.output_dir, "data_as_csv.csv"), index=False)
        aggregates = self.generate_aggregates(processed_df)
        for name, data in aggregates.items():
            if data is not None and not data.empty:
                data.to_csv(os.path.join(self.output_dir, "ESG", f"{name}.csv"), index=False)
            else:
                pd.DataFrame(columns=['date']).to_csv(os.path.join(self.output_dir, "ESG", f"{name}.csv"), index=False)
        
        self._save_connections_and_embeddings(processed_df)
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT ESG Data Fetcher")
    parser.add_argument("--historical", action="store_true", help="Fetch historical data")
    parser.add_argument("--start", type=str, default="20250101", help="Start date (YYYYMMDD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y%m%d"), help="End date (YYYYMMDD)")
    parser.add_argument("--out", type=str, default="Data/GDELT_Live", help="Output directory")
    
    args = parser.parse_args()
    
    fetcher = GDELTFetcher(output_dir=args.out)
    if args.historical:
        fetcher.fetch_historical(args.start, args.end)
    else:
        fetcher.fetch_latest()
