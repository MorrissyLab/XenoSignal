import os
import pandas as pd
import networkx as nx
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def visualize_ipSAE_graph(df: pd.DataFrame, job_name: str, color_map: dict, figures_dir):
    """
    Visualizes a directed graph from a DataFrame with columns 'Chn1', 'Chn2', and 'ipSAE',
    applying node colors and legends from a color_map dictionary.

    Args:
        df (pd.DataFrame): DataFrame with columns 'Chn1', 'Chn2', 'ipSAE'.
        job_name (str): Title and optional filename prefix.
        color_map (dict): Dictionary where keys are group names (e.g., 'Lta'),
                          and values are lists of node names.
                          First group is red, second is blue.

    Returns:
        None. Displays the graph.
    """
    G = nx.DiGraph()

    # Add edges above threshold
    df = df[df["job_name"] == job_name]
    for _, row in df.iterrows():
        if row['ipSAE'] >= 0.17:
            G.add_edge(row['Chn1'], row['Chn2'], weight=1-row['ipSAE'])

    # Assign colors to groups
    group_colors = ['magenta', 'blue']  # extendable
    group_names = list(color_map.keys())
    node_color_dict = {}

    for i, group in enumerate(group_names):
        color = group_colors[i % len(group_colors)]
        for node in color_map[group]:
            node_color_dict[node] = color

    # Use gray for unspecified nodes
    node_colors = [node_color_dict.get(n, 'gray') for n in G.nodes()]

    # Layout
    pos = nx.kamada_kawai_layout(G, weight='weight')
    # pos = nx.circular_layout(G)

    # Edge weights
    edge_weights = [(1-d['weight']) * 5 for _, _, d in G.edges(data=True)]

    # Draw
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        arrows=True,
        edge_color='gray',
        width=edge_weights,
        font_size=12
    )

    # Edge labels
    edge_labels = {(u, v): f"{1-d['weight']:.3f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Create legend
    legend_elements = [
        Patch(facecolor=group_colors[i % len(group_colors)], edgecolor='black', label=group)
        for i, group in enumerate(group_names)
    ]
    plt.legend(handles=legend_elements, loc='best')

    # Show plot
    plt.title(f"ipSAE {job_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"graph_{job_name}.pdf"))
    plt.show()