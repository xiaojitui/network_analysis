import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#from fuzzywuzzy import fuzz
import pickle
import copy

import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt



df = pd.read_csv('./wikipedia/clickstream-enwiki-2019-11.tsv', sep='\t', header=None)
df.columns = ['prev', 'curr', 'type', 'n']


filtered = df[df.curr.notnull() & df.prev.notnull()]
filtered = filtered[(~filtered.curr.str.startswith('other-')) & (~filtered.prev.str.startswith('other-'))]

edges = [(u,v,{'w':d}) for u,v,d in tqdm(filtered[['prev', 'curr', 'n']].values)]

g = nx.DiGraph()
g.add_edges_from(tqdm(edges))

g.remove_node('Main_Page'), g.remove_node('Wikipedia')



#
g.number_of_edges(), g.number_of_nodes()

#
allnodes = list(g.nodes())




clicks = []
paths = []
for name in tqdm(p_nodes):
    others = [k for k in p_nodes if k != name]
    
    for other in others:
        try:
            short_path = nx.shortest_path(g, name, other)
            short_distance = len(short_path) - 1
            clicks.append(short_distance)
            paths.append(short_path)
        except:
            continue
