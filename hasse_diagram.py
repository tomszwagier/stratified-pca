import matplotlib
matplotlib.rc('font', size=20)
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import os

from inference import model_dimension


def build_hasse_diagram(hasse_diagram, root):
    """ Recursive function for building the Hasse diagram of SPCA models. Cf. Figure 3.

    For each model signature, we consider all the sub-signatures obtained by removing
    one element among of the first r-1 dimensions constituting the signature. For
    those sub-signatures, we call recursively the function, until their length is
    equal to 2, which means that we cannot build a simpler flag.
    """
    d = len(root)
    if d > 1:
        for i in range(d - 1):
            child = root[:i] + (root[i] + root[i + 1],) + root[i + 2:]
            hasse_diagram.add_node(child)
            hasse_diagram.add_edge(root, child)
            build_hasse_diagram(hasse_diagram, child)


if __name__ == '__main__':
    np.random.seed(42)
    path = os.path.dirname(__file__) + "/figures/"
    plot_choice = "trajectories"  # original / hierarchical / trajectories

    p = 5
    hasse_diagram = nx.DiGraph()
    root = tuple([1] * p)
    hasse_diagram.add_node(root)
    build_hasse_diagram(hasse_diagram, root)
    pos = graphviz_layout(hasse_diagram, prog="dot")
    pos_labels = dict([(node, (x, y + 14)) for (node, (x, y)) in pos.items()])

    node_colors = [model_dimension(node) for node in hasse_diagram.nodes]
    fig = plt.figure(figsize=(8, 8))
    cmap = plt.cm.coolwarm
    vmin, vmax = min(node_colors), max(node_colors)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    nx.draw_networkx_nodes(hasse_diagram, pos, nodelist=hasse_diagram.nodes, node_color=[sm.to_rgba(model_dimension(node)) for node in hasse_diagram.nodes], node_size=400, edgecolors='black', cmap=cmap)
    nx.draw_networkx_edges(hasse_diagram, pos, width=2, arrows=True, arrowstyle='-', alpha=.2, min_source_margin=0, min_target_margin=0)
    if plot_choice == "trajectories":
        nx.draw_networkx_edges(hasse_diagram, pos, edgelist=[((1, 1, 1, 1, 1), (2, 1, 1, 1)), ((2, 1, 1, 1), (3, 1, 1)), ((3, 1, 1), (4, 1)), ((4, 1), (5,))],
                               width=5.0, edge_color='tab:blue', style='solid', alpha=.8, arrows=True, arrowstyle='-', min_source_margin=0, min_target_margin=0)
        nx.draw_networkx_edges(hasse_diagram, pos, edgelist=[((1, 1, 1, 1, 1), (1, 1, 1, 2)), ((1, 1, 1, 2), (1, 1, 3)), ((1, 1, 3), (1, 4)), ((1, 4), (5,))],
                               width=5.0, edge_color='tab:red', style='solid', alpha=.8, arrows=True, arrowstyle='-', min_source_margin=0, min_target_margin=0)
    if plot_choice == "hierarchical":
        nx.draw_networkx_edges(hasse_diagram, pos, edgelist=[((1, 1, 1, 1, 1), (1, 2, 1, 1)), ((1, 2, 1, 1), (1, 2, 2)), ((1, 2, 2), (3, 2)), ((3, 2), (5,))],
                               width=5.0, edge_color='tab:blue', style='solid', alpha=.8, arrows=True, arrowstyle='->', arrowsize=25, min_source_margin=0, min_target_margin=0)
    if plot_choice in ["original", "trajectories"]:
        nx.draw_networkx_labels(hasse_diagram, pos_labels, None, font_size=16, font_color="black")

    cbar = plt.colorbar(sm, orientation='horizontal', location='bottom', pad=0,
                        fraction=.05, shrink=.8, aspect=50)
    cbar.set_label('Model Dimension')
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(path + f"hasse_{plot_choice}.png", dpi='figure', format='png', transparent=True)
    plt.savefig(path + f"hasse_{plot_choice}.pdf", dpi='figure', format='pdf', transparent=True)
    plt.savefig(path + f"hasse_{plot_choice}.svg", dpi='figure', format='svg', transparent=True)
    plt.close()
