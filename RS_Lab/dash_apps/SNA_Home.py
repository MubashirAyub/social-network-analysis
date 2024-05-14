import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
from dash import html, dcc, callback, Output, Input, State, dash_table
from dash.exceptions import PreventUpdate
from django_plotly_dash import DjangoDash
import pandas as pd
import base64
import datetime
import io
import plotly.graph_objects as go
import powerlaw
import plotly.express as px
import re

np.set_printoptions(threshold=np.inf)

# Global Variables
source = 0
target = 0

# Replace with the file paths of your .edges files
file_path1 = "D://Personal//University_Research_Lab//uni_lab//RS_Lab//dolphin_data.edges"

config = dict({'displaylogo': False, 'displayModeBar': True,
               'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d',
                                          'resetScale2d',
                                          'select2d', 'toggleSpikelines', 'hoverClosestGl2d', 'hoverClosestPie',
                                          'toggleHover',
                                          'resetViews', 'sendDataToCloud', 'toggleSpikelines', 'resetViewMapbox',
                                          'hoverClosestCartesian', 'hoverCompareCartesian'],
               })


# Function to load the dataset from .edges file and create a NetworkX graph
def load_dataset(file_path):
    graph = nx.Graph()
    with open(file_path, "r") as f:
        next(f)
        for line in f:
            parts = re.split(';|,| ', line.strip())
            source_, target_ = map(int, parts[:2])  # Consider only the first two elements (source and target nodes)
            graph.add_edge(source_, target_)

    return graph


def load_file_dataset():
    print('Inside new upload file')

    graph = nx.Graph()
    for i in range(len(source)):
        # print(i)
        graph.add_edge(source[i], target[i])
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
    num_edges_barabasi = max(1,
                             int(num_edges / num_nodes))  # Set a minimum of 1 for the number of edges in Barabási–Albert model
    random_graph = nx.barabasi_albert_graph(num_nodes, num_edges_barabasi)
    random_clustering_coefficient = nx.average_clustering(random_graph)
    random_average_path_length = nx.average_shortest_path_length(random_graph)

    # Check if the network is scale-free using power-law fitting
    fit = powerlaw.Fit(degree_values)
    is_scale_free = fit.power_law.alpha > 2

    # Return the analysis results
    return {
        "Source_Nodes": num_nodes,
        "Target_Nodes": num_edges,
        "Degree Distribution": degree_values,
        "Clustering Coefficient": clustering_coefficient,
        "Average Path Length": average_path_length,
        "Is Small World": clustering_coefficient > random_clustering_coefficient and average_path_length <= random_average_path_length,
        "Is Scale Free": is_scale_free,
        "Assortativity coefficient": Assortivity(graph1),
        "Graph Density": calculate_graph_density(graph1),
        "Closeness Centrality": calculate_closeness_centrality(graph1),
        "Betweenness Centrality": betweenness_centrality(graph1),
        "Eigen Vectors": Eigen_venctors(graph1),
        "Triangles": triangles_value(graph1)
    }


# Function to visualize the degree distribution using Plotly
def plot_degree_distribution(degree_values):
    fig = go.Figure(data=[go.Histogram(x=degree_values, nbinsx=50, histnorm='probability')])
    fig.update_layout(title_text="Degree Distribution", xaxis_title="Degree", yaxis_title="Probability")
    # fig.show()


def Assortivity(G):
    assortativity = nx.degree_assortativity_coefficient(G)
    return assortativity


# def compute_k_core(graph, k):
#     # Initialize the k-core decomposition
#     k_core = graph.copy()
#
#     # Continue removing nodes with degrees less than k until no such nodes exist
#     while True:
#         # Find nodes with degree less than k
#         nodes_to_remove = [node for node in k_core.nodes() if k_core.degree(node) < k]
#
#         # If no such nodes exist, exit the loop
#         if not nodes_to_remove:
#             break
#
#         # Remove the nodes and their incident edges
#         k_core.remove_nodes_from(nodes_to_remove)
#
#     return k_core


def calculate_graph_density(G):
    # Calculate the density of the graph
    graph_density = nx.density(G)

    return graph_density


def calculate_closeness_centrality(G):
    # Calculate the closeness centrality of the graph
    closeness_centrality = nx.closeness_centrality(G)
    # for node, closeness_value in closeness_centrality.items():
    # print(f"Node {node}: Closeness Centrality = {closeness_value}")


def betweenness_centrality(G):
    betweenness_centrality = nx.betweenness_centrality(G)

    # for node, betweenness in betweenness_centrality.items():
    # print(f"Node {node}: Betweenness Centrality = {betweenness}")


def Eigen_venctors(G):
    eigenvector_centrality = nx.eigenvector_centrality(G)

    # for node, eigenvector in eigenvector_centrality.items():
    # print(f"Node {node}: Eigenvector Centrality = {eigenvector}")


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
            size=12,
            colorbar=dict(thickness=15,
                          xanchor='left'),
            line_width=2))

    # Create the graph layout
    layout = go.Layout(
        height=610,
        title_font=dict(
            size=21,
            color='#1f324b'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='closest',
    )

    # Create the triangle count graph figure
    fig = go.Figure(data=[node_trace], layout=layout)


    return fig


def plot_clustering_coefficients_graph(G):
    clustering_coefficients = nx.clustering(G)
    node_ids = list(G.nodes())
    fig = px.scatter(x=node_ids, y=list(clustering_coefficients.values()))

    # Customize the plot
    fig.update_traces(marker=dict(size=10, ))
    fig.update_layout(
        height=610,
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis_title="Node ID",
        yaxis_title="Clustering Coefficient",
    )

    return fig


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
                        height=600,
                        xaxis=dict(title='Degree'),
                        yaxis=dict(title='Count'),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                    ))

    return fig


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
        height=610,
        xaxis=dict(type='log', title='Degree (log scale)'),
        yaxis=dict(type='log', title='Frequency (log scale)'),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    return fig


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

        text = f'Node {node} <br>Connections {len(G.adj[node])} <br>Closeness: {round(closeness, 3)} <br>Betweeness {round(betweenness_cent, 3)} <br>Eigen Vector: {round(eigens, 3)}'
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
                        height=610,
                        title_font=dict(
                            size=21,
                            color='#1f324b'),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


# Load datasets and create network graphs
graph1 = load_dataset(file_path1)

# Analyze the datasets
analysis_results1 = analyze_network(graph1)

FONT_AWESOME = (
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
)

external_stylesheets = [dbc.themes.BOOTSTRAP, FONT_AWESOME]

app = DjangoDash('SNA_Main_APP', external_stylesheets=external_stylesheets)

config = dict({'displaylogo': False, 'displayModeBar': True,
               'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d',
                                          'resetScale2d',
                                          'select2d', 'toggleSpikelines', 'hoverClosestGl2d', 'hoverClosestPie',
                                          'toggleHover',
                                          'resetViews', 'sendDataToCloud', 'toggleSpikelines', 'resetViewMapbox',
                                          'hoverClosestCartesian', 'hoverCompareCartesian'],
               })

if analysis_results1['Is Scale Free']:
    scale_free = "True"
else:
    scale_free = "False"

if analysis_results1['Is Small World']:
    small_world = "True"
else:
    small_world = "False"

## Cards ##
Card_1 = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Network Hypothesis", className="card-title", style={'color': '#c61a27'}),
            html.Hr(),
            html.Br(),
            html.H5("Is Small World : " + small_world, id='small_world_Card'),
            html.H5("Is Scale Free : " + scale_free, id='scale_free_card'),
            html.Br(),
        ]
    ),
    style={"padding": "10px", "margin": "10px"},
)

Card_2 = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Network Insights", className="card-title", style={'color': '#c61a27', 'textAlign': 'Left', }),
            html.Hr(),
            html.Br(),
            html.H5("Nodes : " + str(analysis_results1['Source_Nodes']), id='nodes_g'),
            html.H5("Edges : " + str(analysis_results1['Target_Nodes']), id='edges_g'),
            html.H5("Clustering Coefficient : " + str(analysis_results1['Clustering Coefficient']),
                    id='cluster_coe_card'),
            html.H5("Average Path Length : " + str(analysis_results1['Average Path Length']), id='path_len_card'),
            html.H5("Assortativity coefficient : " + str(analysis_results1['Assortativity coefficient']),
                    id='assortivity_card'),
            html.H5("Graph Density : " + str(analysis_results1['Graph Density']), id='density_card'),
            html.H5("Triangles : " + str(analysis_results1['Triangles']), id='tri_card'),
            html.Br(),
        ]
    ),
    style={"padding": "10px", "margin": "10px"},
)

cards = dbc.CardGroup(
    [
        Card_1,
        Card_2,
    ],
    className='text-left'
)

row = html.Div(
    [
        dbc.Row(
            [
                html.H5('Please upload network file',
                        style={'textAlign': 'left', 'color': '#c61a27',
                               }),
                html.P('*Rename column names to : "Source_Nodes" and "Target_Nodes"', style={'textAlign': 'left', 'color': '#c61a27',
                               }),
                dbc.Col(
                    html.Div([
                        dcc.Upload(
                            id='upload_data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),
                        html.Div(id='output_data_upload'),
                        html.Div([
                            html.Button('Submit', id='submit-val', n_clicks=0, style={'color': '#c61a27'}),
                        ], style={'text-align': 'right', "align": "center", }, ),

                    ]),
                ),
            ],
            align="end", style={"margin": "15px", 'margin-bottom': '40px', "padding": "15px", "background": "#F1F1F1",
                                'border-top': '10px solid #F1F1F1', 'border-bottom': '10px solid #F1F1F1'},
        ),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3('Social Network Analysis',
                            style={'textAlign': 'Center', 'color': '#1f324b', 'margin-bottom': '30px',
                                   'margin-top': '20px',
                                   'padding': '10px'}),
                    html.Div([
                        html.H5('Sample Data Set:', id='sample_heading', style={'color': '#c61a27'}),
                        html.H6(
                            '1. A social network of bottlenose dolphins. The dataset contains a list of all of links, where a link represents frequent associations between dolphins',
                            id='sample_text'),
                        html.H5('Data Citation', style={'color': '#c61a27'}, id='sample_citation'),
                        html.H6(
                            'Title: The Network Data Repository with Interactive Graph Analytics and Visualization',
                            id="sample_cita_1"),
                        html.H6('Author: Ryan A. Rossi and Nesreen K. Ahmed', id="sample_cita_2"),
                        html.H6('URL: https://networkrepository.com', id="sample_cita_3"),
                    ]),

                    cards,
                ],
                    style={"margin": "15px", 'margin-bottom': '20px', "padding": "15px", "background": "#F1F1F1",
                           'border-top': '5px solid #F1F1F1', 'border-bottom': '5px solid #F1F1F1'},
                ),
            ])
        ]),

        dbc.Row(
            [
                html.H5('Degree Distribution (log-log plot)',
                        style={'textAlign': 'left', 'color': '#c61a27',
                               }),
                dbc.Col(html.Div(dcc.Graph(id='fig_Powerlog', figure=power_law_loglogplot(graph1), config=config)),
                        width=12,
                        align="start"),
            ],
            align="end", style={"margin": "15px", 'margin-bottom': '40px', "padding": "15px", "background": "#F1F1F1",
                                'border-top': '10px solid #F1F1F1', 'border-bottom': '10px solid #F1F1F1'},
        ),

        dbc.Row(
            [
                html.H5('Network Graph',
                        style={'textAlign': 'left', 'color': '#c61a27', }),
                dbc.Col(html.Div(dcc.Graph(id='fig_Network_plot', figure=draw_network_diagram(graph1), config=config)),
                        width=12,
                        align="start"),
            ],
            align="end", style={"margin": "15px", 'margin-bottom': '40px', "padding": "15px", "background": "#F1F1F1",
                                'border-top': '10px solid #F1F1F1', 'border-bottom': '10px solid #F1F1F1'},
        ),

        dbc.Row(
            [
                html.H5('Clustering Coefficients for Nodes',
                        style={'textAlign': 'left', 'color': '#c61a27', }),
                dbc.Col(html.Div(dcc.Graph(id='fig_clus_coef_plot', figure=plot_clustering_coefficients_graph(graph1),
                                           config=config)), width=12,
                        align="start"),

            ],
            align="end", style={"margin": "15px", 'margin-bottom': '40px', "padding": "15px", "background": "#F1F1F1",
                                'border-top': '10px solid #F1F1F1', 'border-bottom': '10px solid #F1F1F1'},
        ),

        dbc.Row(
            [
                html.H5('Triangle Count',
                        style={'textAlign': 'left', 'color': '#c61a27',
                               }),
                dbc.Col(html.Div(dcc.Graph(id='fig_tricount_plot', figure=draw_triangle_graph(graph1), config=config)),
                        width=12,
                        align="start"),

            ],
            align="end", style={"margin": "15px", 'margin-bottom': '40px', "padding": "15px", "background": "#F1F1F1",
                                'border-top': '10px solid #F1F1F1', 'border-bottom': '10px solid #F1F1F1'},
        ),
    ])

# Dash Application Layout
app.layout = dbc.Container([
    html.H1('Koblenz University Research Lab Project',
            style={'textAlign': 'Center', 'color': '#1f324b', 'margin-bottom': '30px',
                   'margin-top': '20px', 'border-bottom': '2px dashed', 'border-bottom-color': '#1f324b',
                   'padding': '15px'}),
    row,
], fluid=True,
)


def parse_contents(contents, filename, date):
    global source, target, analysis_results1, graph1
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:

            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

            source = df['Source_Nodes'].tolist()
            target = df['Target_Nodes'].tolist()

            graph1 = load_file_dataset()
            analysis_results1 = analyze_network(graph1)


        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            style_data={
                'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_table={'overflowX': 'scroll'},
            page_action='native',
            page_current=0,
            page_size=11,
        ),
    ])


@app.callback(
    [
        Output(component_id='output_data_upload', component_property='children'),

    ],
    [
        Input(component_id='upload_data', component_property='contents'),
        State(component_id='upload_data', component_property='last_modified'),
        State(component_id='upload_data', component_property='filename')
    ]
)
def update_drop_down_Firms(list_of_contents, list_of_dates, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    [
        Output('nodes_g', 'children'),
        Output('edges_g', 'children'),
        Output('cluster_coe_card', 'children'),
        Output('path_len_card', 'children'),
        Output('assortivity_card', 'children'),
        Output('density_card', 'children'),
        Output('tri_card', 'children'),
        Output('small_world_Card', 'children'),
        Output('scale_free_card', 'children'),
        Output(component_id='fig_Powerlog', component_property='figure'),
        Output(component_id='fig_Network_plot', component_property='figure'),
        Output(component_id='fig_clus_coef_plot', component_property='figure'),
        Output(component_id='fig_tricount_plot', component_property='figure'),
        Output('sample_heading', 'children'),
        Output('sample_text', 'children'),
        Output('sample_cita_1', 'children'),
        Output('sample_cita_2', 'children'),
        Output('sample_cita_3', 'children'),
        Output('sample_citation', 'children'),

    ],
    [
        Input('submit-val', 'n_clicks'),
    ], prevent_initial_call=True
)
def update_cards(n_clicks):
    scale_free = 'False'
    small_world = 'False'
    if analysis_results1['Is Scale Free']:
        scale_free = "True"
    else:
        scale_free = "False"

    if analysis_results1['Is Small World']:
        small_world = "True"
    else:
        small_world = "False"

    nodes_val = "Nodes : " + str(analysis_results1['Source_Nodes'])
    edges_val = "Edges : " + str(analysis_results1['Target_Nodes'])
    cluster_coe_val = "Clustering Coefficient : " + str(analysis_results1['Clustering Coefficient'])
    path_len_val = "Average Path Leng : " + str(analysis_results1['Average Path Length'])
    assor_val = "Assortativity coefficient : " + str(analysis_results1['Assortativity coefficient'])
    g_density_val = "Graph Density : " + str(analysis_results1['Graph Density'])
    tri_val = "Triangles : " + str(analysis_results1['Triangles'])
    small_world_val = "Is Small World : " + str(small_world)
    Scale_free_val = "Is Scale Free : " + str(scale_free)
    powerlog_fig = power_law_loglogplot(graph1)
    network_plot_fig = draw_network_diagram(graph1)
    clus_coef_fig = plot_clustering_coefficients_graph(graph1)
    tri_count_Fig = draw_triangle_graph(graph1)


    return [nodes_val, edges_val, cluster_coe_val, path_len_val, assor_val, g_density_val, tri_val, small_world_val, Scale_free_val, powerlog_fig, network_plot_fig, clus_coef_fig, tri_count_Fig, '', '', '', '', '', '']
