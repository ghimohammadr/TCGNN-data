from torch_geometric.datasets import Planetoid, LastFMAsia, CitationFull, AttributedGraphDataset, WebKB, Actor, WikipediaNetwork
import torch_geometric.transforms as T
import torch
from torch import nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_dense_adj, coalesce, remove_self_loops, to_undirected
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import os

# read the data
def load_data(name_data, device):

    # # Load the existing graph data from the pickle file (NetworkX graph)
    # with open('melbourne_graph1.pkl', 'rb') as f:
    #     G = pickle.load(f)
    # Define the path to the pickle file in the 'download' folder
    file_path = os.path.join('Datasets', 'melbourne_graph1.pkl')

    # Load the existing graph data from the pickle file (NetworkX graph)
    with open(file_path, 'rb') as f:
        G = pickle.load(f)

    # Check the graph type
    print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

    # Define the bounding box for the area you want to select (Melbourne area example)
    north, south, east, west = -37.800, -37.810, 145.400, 144.800  # Smaller area in Melbourne
    print(f"Filtering nodes within bounding box: North={north}, South={south}, East={east}, West={west}")

    # Create a list to store the filtered nodes within the bounding box
    filtered_nodes = []

    # Loop through each node in the graph and check if it's within the bounding box
    for node, data in G.nodes(data=True):  # Looping through all nodes
        lat, lon = data['y'], data['x']  # Assuming 'y' is lat and 'x' is lon
        if south <= lat <= north and west <= lon <= east:
            filtered_nodes.append(node)

    # Create a subgraph with only the filtered nodes
    subgraph_edges = [(u, v) for u, v in G.edges() if u in filtered_nodes and v in filtered_nodes]

    # Create the subgraph with the filtered nodes and edges
    G = G.subgraph(filtered_nodes)

    G = G.to_undirected()


    # 2. Convert to a simple graph (removes multiple edges if necessary)
    G = nx.Graph(G)  # This removes multiple edges between nodes

    # 3. Relabel nodes to integer indices
    G = nx.convert_node_labels_to_integers(G)

    # 4. Extract edge index (PyG style)
    # edge_index = torch.tensor(list(G.edges)).t().contiguous()
    # import torch

    # Assuming you have a list of edges in subgraph_edges and filtered_nodes
    # Create a mapping from the original nodes to the subgraph's new indices
    node_mapping = {node: idx for idx, node in enumerate(filtered_nodes)}

    # Remap the subgraph edges
    remapped_edges = [(node_mapping[u], node_mapping[v]) for u, v in subgraph_edges]

    # Now create the edge_index tensor from the remapped edges
    edge_index = torch.tensor(remapped_edges).t().contiguous()




    # 5. Node features (example: degree, clustering coefficient)
    degrees = np.array([val for (node, val) in G.degree()])
    ones = np.array([1 for (node, val) in G.degree()])
    clustering = np.array(list(nx.clustering(G).values()))  # Now works after converting to simple graph
    # node_feats = np.vstack((degrees, clustering)).T
    node_feats = np.vstack((ones))

    x = torch.tensor(node_feats, dtype=torch.float)

    # 6. Edge features (e.g., length, highway type one-hot)
    lengths = []
    highway_types = []

    # Collecting edge data
    for u, v, data in G.edges(data=True):
        lengths.append(data.get('length', 1.0))  # Default to 1.0 if length is missing
        highway_type = data.get('highway', 'unknown')  # Replace missing highway with 'unknown'
        
        # Ensure each highway_type is a simple string
        if isinstance(highway_type, str):
            highway_types.append(highway_type)
        else:
            highway_types.append('unknown')  # Handle non-string highway types by replacing with 'unknown'

    # Convert highway_types to a numpy array
    highway_types = np.array(highway_types, dtype=str)

    # One-hot encode highway type
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    highway_encoded = encoder.fit_transform(highway_types.reshape(-1, 1))

    # Combine length and encoded highway types
    edge_attr = torch.tensor(np.column_stack([lengths, highway_encoded]), dtype=torch.float)

    # # Assign labels based on synthetic importance (e.g., betweenness centrality ranking)
    betweenness = nx.betweenness_centrality(G)
    y = torch.tensor([1 if betweenness[i] > np.median(list(betweenness.values())) else 0 for i in range(np.shape(x)[0])], dtype=torch.long)


    # Create the PyG Data object for the subgraph
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    print(data)



    # Find unique nodes in the edge_index
    unique_nodes = torch.unique(data.edge_index)

    # Get the number of unique nodes
    num_nodes = unique_nodes.size(0)
    print(num_nodes)


    # Assuming data.edge_index is the edge index of the graph
    edge_index = data.edge_index

    # Create a sparse tensor from edge_index
    # Create an identity tensor for the values (1.0 for each edge)
    values = torch.ones(edge_index.size(1))

    # Create a sparse tensor
    adj_matrix_sparse = torch.sparse_coo_tensor(edge_index, values, (data.num_nodes, data.num_nodes))

    adj_matrix = torch.squeeze(adj_matrix_sparse.to_dense())

    n_classes = 2
    num_features = np.shape(x)[1]

    return data, adj_matrix, n_classes, num_features
