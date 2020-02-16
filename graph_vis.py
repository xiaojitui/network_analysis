#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:





# In[ ]:





# In[2]:


def get_edges(map_df, nodes, others):
    
    edges = []
    
    for node in nodes:
        
        for other in others:
            
            try:

                path = map_df.loc[node, other]
            
                if isinstance(path, float):
                    continue

                path = eval(path)
                for i in range(len(path) - 1):

                    edge = (path[i], path[i+1])

                    if edge not in edges:
                        edges.append(edge)
                        
            except:
                continue
        
    return edges


# In[3]:


def get_edges_1(map_df, nodes, others, thresh):
    
    edges = []
    
    for node in nodes:
        
        for other in others:
            
            try:

                path = map_df.loc[node, other]
            
                if isinstance(path, float):
                    continue

                path = eval(path)
                
                if len(path) - 1 > thresh:
                    continue
                
                for i in range(len(path) - 1):

                    edge = (path[i], path[i+1])

                    if edge not in edges:
                        edges.append(edge)
                        
            except:
                continue
        
    return edges


# In[4]:


def get_node_size(g, c_node):
    
    
    node_sizes = []
    
    allnodes = list(g.nodes())
    
    for node in allnodes:
        
        if node == c_node:
            node_sizes.append(2500)
            continue
            
        try: 
            from_n = nx.shortest_path_length(g, c_node, node)
        except:
            from_n = np.inf
            
        try: 
            to_n = nx.shortest_path_length(g, node, c_node)
        except:
            to_n = np.inf
        
        node_sizes.append(500/min(from_n, to_n))
            
    return node_sizes


# In[5]:


def get_node_size_1(g, c_node, name_nodes):
    
    
    node_sizes = []
    
    allnodes = list(g.nodes())
    
    for node in allnodes:
        
        if node == c_node:
            node_sizes.append(2500)
            #continue
            
        elif node in name_nodes:
            node_sizes.append(500)
            #continue
            
        else:
            node_sizes.append(100)
            
    return node_sizes


# In[6]:


def get_node_color(g, c_node):
    
    
    node_color = []
    
    allnodes = list(g.nodes())
    
    for node in allnodes:
        
        if node == c_node:
            node_color.append('red')
            continue
            
        try: 
            from_n = nx.shortest_path_length(g, c_node, node)
        except:
            from_n = np.inf
            
        try: 
            to_n = nx.shortest_path_length(g, node, c_node)
        except:
            to_n = np.inf
            
        if from_n == 1 or to_n == 1:
            node_color.append('green')
        else:
            node_color.append('blue')
            
    return node_color


# In[7]:


def get_node_color_1(g, c_node, namelist):
    
    
    node_color = []
    
    allnodes = list(g.nodes())
    
    for node in allnodes:
        
        if node == c_node:
            node_color.append('red')
            continue
            
        if node in namelist: 
            node_color.append('blue')
        else:
            node_color.append('k')
            
    return node_color


# In[8]:


def get_node_pos(map_df, c_name, ratio = 0.5):
    
    name = c_name
    c_node = name_map[name]
    
    pos_record = []
    for i in range(10):
        edges1 = get_edges_1(map_df, [name], map_df.columns.to_list(), thresh = i+1)
        edges2 = get_edges_1(map_df, map_df.index.to_list(), [name], thresh = i+1)
        edges = edges1 + edges2
        
        d = nx.MultiDiGraph()
        _ = d.add_edges_from(tqdm(edges))
        
        lo = nx.circular_layout(d, ratio*(i+1))
        lo[c_node] = np.array([0, 0])
        pos_record.append(lo)
        
    return pos_record


# In[9]:


def update_pos(pos_record):
    eles = copy.deepcopy(pos_record)
    
    ele = eles[-1]
    for i in range(len(eles)-2, -1, -1):
        ele.update(eles[i])
    
    return ele


# In[ ]:





# In[ ]:





# In[ ]:


## test

if __name__ == '__main__':
	map_df = pd.read_csv('./graph.csv', index_col=0)
	## this is a M*M matrix, cell[i, j] = path from i to j 
	## to generate map_df, do:
	#### g = nx.DiGraph()
	#### g.add_edges_from(tqdm(edges))
	#### nx.shortest_path(g, node1, node2)


	# In[ ]:


	# c_name is the center 
	pos_record = get_node_pos(map_df, c_name)
	lo = update_pos(pos_record)



	#name = map_df.index.to_list()[100]


	## get 2 layers
	name = c_name
	edges1 = get_edges_1(map_df, [name], map_df.columns.to_list(), thresh = 2)
	edges2 = get_edges_1(map_df, map_df.index.to_list(), [name], thresh = 2)

	edges = edges1 + edges2


	d = nx.MultiDiGraph()
	_ = d.add_edges_from(tqdm(edges))



	# only show nodes we are interested in
	name_nodes = [k for k in d.nodes() if k in name_map_r]
	labels = {}

	for ele in name_nodes:
		k = name_map_r[ele][0]
		
		labels[ele]  = k


	# In[ ]:


	#edge_labels=dict([((u,v,),d['text']) for u,v,d in dg1.edges.data()])

	fig, ax = plt.subplots(1, 1, figsize = (20, 20))
	ax.set_title('Network Graph: "%s"' % name, fontdict = {'fontsize': 24})


	#labels = {k:k for k in list(d.nodes())}
	c_node = name_map[name]
	#node_sizes = [100*d.degree[node] for node in d.nodes()]
	#node_sizes = [400/nx.shortest_path_length(d, c_node, node) for node in list(d.nodes())[1:]]
	#node_sizes = [max(node_sizes) * 4] + node_sizes

	#node_sizes = get_node_size(d, c_node)
	node_sizes = get_node_size_1(d, c_node, name_nodes)

	#lo = nx.spring_layout(d, pos = {c_node: [0, 0]}, fixed = [c_node], k = 1)# 

	#lo = nx.circular_layout(d, 2)
	#lo[c_node] = np.array([0, 0])
	_edges = nx.draw_networkx_edges(d, lo, ax = ax, width = 0.4, alpha = 0.5)
	_nodes = nx.draw_networkx_nodes(d, lo, ax = ax, node_size=node_sizes, alpha = 0.6, 
									node_color = get_node_color_1(d, c_node, name_nodes))
									#node_color= ['red'] + ['blue'] * (d.number_of_nodes() - 1))

	_labels = nx.draw_networkx_labels(d, lo, labels)
	# _edges = nx.draw_networkx_edge_labels(d, lo, alpha = 0.5, edge_labels=edge_labels)
	# _edges = nx.draw_networkx_edges(d, lo, ax = ax, width = 0.4, alpha = 0.5)

	# nx.draw_networkx(d, lo, ax = ax, node_size=node_sizes, with_labels=True)


	# In[ ]:




