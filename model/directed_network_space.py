from mesa.space import NetworkGrid

class DirectedNetworkSpace(NetworkGrid):
    """Network Space where each node contains zero or more agents and where each edge is directed."""
    def get_neighbors(self, node_id: int, include_center: bool = False, directed: bool = False) -> List[int]:
        """Get all adjacent nodes"""

        neighbors = []
        if directed:
            for edge in self.G.edges:
                if edge[0] == node_id:
                    neighbors.append(edge[1])
        else:
            neighbors = list(self.G.neighbors(node_id))
            if include_center:
                neighbors.append(node_id)

        return neighbors