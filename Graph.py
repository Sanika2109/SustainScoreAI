import pandas as pd
import numpy as np
import networkx as nx
import itertools

import pandas as pd
import numpy as np
import networkx as nx
import itertools

class graph_creator:
    def __init__(self, df):
        self.df = df

    def create_graph(self):

        # ✅ MOVE VALIDATION HERE
        if self.df is None or self.df.empty:
            return nx.Graph()

        if "URL" not in self.df.columns or "Organization" not in self.df.columns:
            raise ValueError("DataFrame must contain 'URL' and 'Organization'")

        # Find Edges
        df_edge = (
            self.df.groupby("URL")["Organization"]
            .apply(lambda x: list(set([str(i).strip().upper() for i in x if pd.notna(i)])))
            .reset_index()
        )

        def get_tpls(r):
            if len(r) > 1:
                return [(a, b) for a, b in itertools.combinations(r, 2) if a != b]
            return None

        df_edge["SourceDest"] = df_edge.Organization.apply(get_tpls)
        df_edge = df_edge.explode("SourceDest")

        # Get Weights
        source_dest = pd.DataFrame(
            df_edge.SourceDest.tolist(),
            columns=["Source", "Dest"]
        )

        source_dest = source_dest.groupby(
            ["Source", "Dest"]
        ).size().reset_index(name="weight")
        
        if not source_dest.empty:
            source_dest["weight"] = source_dest["weight"] / source_dest["weight"].max()


        # Build Graph
        self.organizations = set(source_dest.Source).union(set(source_dest.Dest))

        self.G = nx.from_pandas_edgelist(
            source_dest,
            source="Source",
            target="Dest",
            edge_attr="weight",
            create_using=nx.Graph
        )

        return self.G
