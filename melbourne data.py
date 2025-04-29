import pickle
import osmnx as ox
import torch
from torch_geometric.data import Data

# Load the existing graph data from the pickle file (NetworkX graph)
with open('melbourne_graph1.pkl', 'rb') as f:
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

# Now extract the corresponding node features for the filtered nodes
# For simplicity, you can use the degree of each node as the node feature
node_feats = torch.tensor([[deg] for node, deg in G.degree(filtered_nodes)], dtype=torch.float)

# Optionally, you may need to update edge attributes based on the filtered subgraph (e.g., lengths, types)
edge_attr = torch.tensor([G[u][v].get('length', 1.0) for u, v in subgraph_edges], dtype=torch.float)

# Create the PyG Data object for the subgraph
subgraph_data = Data(
    x=node_feats,
    edge_index=torch.tensor(subgraph_edges).t().contiguous(),
    edge_attr=edge_attr  # Add edge attributes if necessary
)

print(subgraph_data)
# Optionally, you can add node labels (if they exist)
# subgraph_data.y = node_labels_for_subgraph  # If you have labels, add them here

# Save the filtered subgraph data to a new pickle file
with open('melbourne_subgraph_data.pkl', 'wb') as f:
    pickle.dump(subgraph_data, f)

print(f"Subgraph data with {len(filtered_nodes)} nodes and {len(subgraph_edges)} edges has been saved.")
