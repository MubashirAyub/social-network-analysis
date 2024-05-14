import os
from collections import defaultdict
import plotly.express as px
import networkx as nx
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import powerlaw

# Function to load the dataset from .edges file and create a NetworkX graph
def load_dataset(file_path):
    graph = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            source, target = map(int, parts[:2])  # Consider only the first two elements (source and target nodes)
            graph.add_edge(source, target)
    return graph

def analyze_network(graph):
    # Get the largest connected component
    largest_component = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_component)

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    degree_values = np.array(list(dict(graph.degree()).values()))

    # Calculate clustering coefficient and average path length for comparison with random graphs
    clustering_coefficient = nx.average_clustering(graph)
    average_path_length = nx.average_shortest_path_length(graph)

    # Check if the network is small-world
    num_edges_barabasi = max(1, int(num_edges / num_nodes))  # Set a minimum of 1 for the number of edges in Barabási–Albert model
    random_graph = nx.barabasi_albert_graph(num_nodes, num_edges_barabasi)
    random_clustering_coefficient = nx.average_clustering(random_graph)
    random_average_path_length = nx.average_shortest_path_length(random_graph)

    # Check if the network is scale-free using power-law fitting
    fit = powerlaw.Fit(degree_values)
    is_scale_free = fit.power_law.alpha > 2

    # Return the analysis results
    return {
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Degree Distribution": degree_values,
        "Clustering Coefficient": clustering_coefficient,
        "Average Path Length": average_path_length,
        "Is Small World": clustering_coefficient > random_clustering_coefficient and average_path_length <= random_average_path_length,
        "Is Scale Free": is_scale_free,
        "Assortativity coefficient" : Assortivity(graph1),
        "Graph Density" : calculate_graph_density(graph1),
        "Closeness Centrality" : calculate_closeness_centrality(graph1),
        "Betweenness Centrality" : betweenness_centrality(graph1),
        "Eigen Vectors" : Eigen_venctors(graph1),
        "Triangles" : triangles_value(graph1),
        "K Core" : compute_k_core(graph1, 1),
        # "Adjency Metrix" : draw_adjency_metrix(graph1)
    }

# Function to visualize the degree distribution using Plotly
def plot_degree_distribution(degree_values):
    fig = go.Figure(data=[go.Histogram(x=degree_values, nbinsx=50, histnorm='probability')])
    fig.update_layout(title_text="Degree Distribution", xaxis_title="Degree", yaxis_title="Probability")
    fig.show()


def plot_clustering_coefficients_graph(G):
    clustering_coefficients = nx.clustering(G)
    node_ids = list(G.nodes())
    fig = px.scatter(x=node_ids, y=list(clustering_coefficients.values()))

    # Customize the plot
    fig.update_traces(marker=dict(size=10, opacity=0.6))
    fig.update_layout(
        title="Clustering Coefficients for Nodes",
        xaxis_title="Node ID",
        yaxis_title="Clustering Coefficient",
    )

    # Show the plot
    fig.show()

def draw_network_diagram(G):

    pos = nx.spring_layout(G)

    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_closeness = []
    node_betweenness = []
    node_eigenvector = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        closeness = nx.closeness_centrality(G, node)
        betweenness = nx.betweenness_centrality(G, node)
        eigenvector = nx.eigenvector_centrality(G)
        betweenness_cent = betweenness[node]
        eigens = eigenvector[node]

        node_closeness.append(closeness)
        node_betweenness.append(betweenness)

        text = f'Node {node} <br>Connections {len(G.adj[node])} <br>Closeness: {round(closeness,3)} <br>Betweeness {round(betweenness_cent,3)} <br>Eigen Vector: {round(eigens,3)}'
        node_text.append(text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_closeness,  # Change this to the desired centrality measure
            size=10,
            colorbar=dict(
                thickness=15,
                title='Centrality',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()



def Assortivity(G):

    assortativity = nx.degree_assortativity_coefficient(G)
    return assortativity

def compute_k_core(graph, k):
    k_core = graph.copy()

    while True:
        nodes_to_remove = [node for node in k_core.nodes() if k_core.degree(node) < k]

        if not nodes_to_remove:
            break

        k_core.remove_nodes_from(nodes_to_remove)

    return k_core.nodes()



def calculate_graph_density(G):

    graph_density = nx.density(G)

    return graph_density


def calculate_closeness_centrality(G):

    closeness_centrality = nx.closeness_centrality(G)
    for node, closeness_value in closeness_centrality.items():
        print(f"Node {node}: Closeness Centrality = {closeness_value}")

def betweenness_centrality(G):
    print("--------------------------------------------------------------------")
    betweenness_centrality = nx.betweenness_centrality(G)

    for node, betweenness in betweenness_centrality.items():
        print(f"Node {node}: Betweenness Centrality = {betweenness}")


def Eigen_venctors(G):

    eigenvector_centrality = nx.eigenvector_centrality(G)

    for node, eigenvector in eigenvector_centrality.items():
        print(f"Node {node}: Eigenvector Centrality = {eigenvector}")

def triangles_value(G):
     num_triangles = sum(nx.triangles(G).values()) // 3
     return num_triangles



def draw_triangle_graph(G):
    triangles = list(nx.enumerate_all_cliques(G))  # Enumerate all triangles

    # Create a dictionary to store the triangle count for each node
    triangle_count = {node: 0 for node in G.nodes()}

    # Count the triangles each node participates in
    for triangle in triangles:
        for node in triangle:
            triangle_count[node] += 1

    # Layout the graph
    pos = nx.spring_layout(G)

    # Create a scatter plot for nodes with triangle count as hovertext
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    triangle_counts = [triangle_count[node] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=triangle_counts,  # This sets the hovertext as the triangle count
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            colorbar=dict(thickness=15, title='Triangle Count', xanchor='left'),
            line_width=2))

    # Create the graph layout
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'))

    # Create the triangle count graph figure
    fig = go.Figure(data=[node_trace], layout=layout)

    # Show the triangle count graph
    fig.show()

def draw_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = {}
    for degree in degree_sequence:
        if degree in degree_count:
            degree_count[degree] += 1
        else:
            degree_count[degree] = 1

    # Create a histogram of the degree distribution
    degrees = list(degree_count.keys())
    counts = list(degree_count.values())

    # Create a line graph for the degree distribution
    fig = go.Figure(data=[go.Scatter(x=degrees, y=counts, mode='lines')],
                    layout=go.Layout(
                        title='Degree Distribution',
                        xaxis=dict(title='Degree'),
                        yaxis=dict(title='Count'),
                    ))

    # Show the line graph
    fig.show()


def draw_k_core(G):
    k_core = nx.k_core(G)

    # Define node positions (you can use different layouts)
    pos = nx.spring_layout(G)

    # Create a scatter plot for nodes with different colors for each k-core
    node_trace = []
    for core_number in range(max(k_core.values()) + 1):
        nodes_in_core = [node for node, core in k_core.items() if core == core_number]
        x = [pos[node][0] for node in nodes_in_core]
        y = [pos[node][1] for node in nodes_in_core]

        node_trace.append(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=8,
                opacity=0.6,
                color=core_number,  # Use the core number as a color
                colorbar=dict(
                    thickness=15,
                    title=f'K-Core {core_number}',
                    xanchor='left',
                    titleside='right'
                ),
                colorscale='Viridis'
            ),
            text=[f'Node {node}' for node in nodes_in_core]
        ))

    # Create the graph layout
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Create the k-core decomposition visualization figure
    fig = go.Figure(data=node_trace, layout=layout)

    # Show the k-core decomposition visualization
    fig.show()

def power_law_loglogplot(G):
    # Calculate the degree distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_counts = nx.degree_histogram(G)

    # Create a log-log plot
    x = [i for i, count in enumerate(degree_counts) if count > 0]
    y = [count for count in degree_counts if count > 0]

    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=8),
        text=[f'Degree {i}' for i in x],
    ))

    fig.update_layout(
        xaxis=dict(type='log', title='Degree (log scale)'),
        yaxis=dict(type='log', title='Frequency (log scale)'),
        title='Degree Distribution (log-log plot)',
    )

    fig.show()

# Replace with the file paths of your .edges files
file_path1 = "dolphin_data.edges"

# Load datasets and create network graphs
graph1 = load_dataset(file_path1)

# Analyze the datasets
analysis_results1 = analyze_network(graph1)

# Visualize the degree distribution of each dataset using Plotly
# plot_degree_distribution(analysis_results1["Degree Distribution"])
# plot_clustering_coefficients_graph(graph1)
draw_network_diagram(graph1)
# draw_triangle_graph(graph1)
# draw_degree_distribution(graph1)
# draw_k_core(graph1)
power_law_loglogplot(graph1)

# Print analysis results for each dataset
print("Dataset 1 Analysis:")
print(analysis_results1)




