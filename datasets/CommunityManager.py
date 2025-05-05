import os
import logging
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt

class CommunityManager:
    def __init__(self, datasets_root='datasets', edge_filename='mapped_nodes_edges.txt', seed=42, overwrite=False):
        self.datasets_root = datasets_root
        self.edge_filename = edge_filename
        self.seed = seed
        self.overwrite = overwrite
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def process_all(self):
        for dirpath, dirnames, filenames in os.walk(self.datasets_root):
            if self.edge_filename in filenames:
                edge_path = os.path.join(dirpath, self.edge_filename)
                save_node_path = os.path.join(dirpath, 'node_to_community.csv')
                save_comm_path = os.path.join(dirpath, 'communities.txt')
                vis_path = os.path.join(dirpath, 'community_visualization.png')

                if not self.overwrite and os.path.exists(save_node_path) and os.path.exists(save_comm_path):
                    self.logger.info(f"⏩ Skipping (already exists): {dirpath}")
                    continue

                self.logger.info(f"🔍 Processing: {edge_path}")
                try:
                    G_nx = nx.read_edgelist(edge_path, nodetype=int)
                    if len(G_nx) == 0:
                        self.logger.warning(f"⚠️ Empty graph: {edge_path}")
                        continue
                except Exception as e:
                    self.logger.error(f"❌ Failed to read graph: {e}")
                    continue

                try:
                    communities = self._leiden_communities(G_nx)
                    node_to_comm = {
                        node: i for i, comm in enumerate(communities) for node in comm
                    }

                    # 保存映射表
                    df = pd.DataFrame(list(node_to_comm.items()), columns=['node', 'community'])
                    df.to_csv(save_node_path, index=False)

                    # 保存社区节点列表
                    with open(save_comm_path, 'w') as f:
                        for i, comm in enumerate(communities):
                            f.write(f'Community {i}: {",".join(map(str, comm))}\n')

                    self._plot_community_graph(G_nx, node_to_comm, vis_path)

                    # 打印统计信息
                    num_comms = len(communities)
                    avg_size = sum(len(c) for c in communities) / num_comms
                    self.logger.info(f"✅ Communities found: {num_comms} (avg size: {avg_size:.1f})")
                    self.logger.info(f"🖼️  Saved visualization: {vis_path}\n")

                except Exception as e:
                    self.logger.error(f"❌ Error in community detection: {e}")

    def _leiden_communities(self, G_nx):
        G_ig, reverse_map = self._nx_to_igraph(G_nx)
        partition = leidenalg.find_partition(
            G_ig,
            leidenalg.ModularityVertexPartition,
            seed=self.seed
        )
        communities = []
        for comm in partition:
            nodes = [reverse_map[i] for i in comm]
            communities.append(set(nodes))
        return communities

    def _nx_to_igraph(self, G_nx):
        mapping = dict(zip(G_nx.nodes(), range(G_nx.number_of_nodes())))
        reverse_mapping = {v: k for k, v in mapping.items()}
        G_nx = nx.relabel_nodes(G_nx, mapping)
        G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)
        return G_ig, reverse_mapping

    def _plot_community_graph(self, G_nx, node_to_comm, save_path):
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G_nx, seed=self.seed)
        communities = set(node_to_comm.values())
        num_comms = len(communities)
        colors = plt.get_cmap('tab20')(range(num_comms))
        comm_to_color = {comm: colors[i % len(colors)] for i, comm in enumerate(communities)}
        node_colors = [comm_to_color[node_to_comm[n]] for n in G_nx.nodes()]

        nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, node_size=30, alpha=0.8)
        nx.draw_networkx_edges(G_nx, pos, alpha=0.2, width=0.3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()

# ✅ 支持直接运行
if __name__ == '__main__':
    manager = CommunityManager(datasets_root='datasets', edge_filename='mapped_nodes_edges.txt', overwrite=True)
    manager.process_all()
