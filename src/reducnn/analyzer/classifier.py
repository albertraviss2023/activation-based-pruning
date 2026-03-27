from typing import Dict, Any, List, Optional
from ..core.adapter import FrameworkAdapter

class ArchitectureClassifier:
    """Analyzes model topology to identify interdependent layer clusters.
    
    This classifier uses the framework-agnostic graph representation to 
    group layers into 'Pruning Clusters'. Layers in the same cluster 
    must be pruned using identical masks to maintain structural compatibility
    (e.g., layers feeding into an addition node). Concatenation nodes 
    do not create clusters as their inputs can be pruned independently.
    """
    
    def __init__(self, adapter: FrameworkAdapter):
        """Initializes the classifier with a framework adapter.
        
        Args:
            adapter (FrameworkAdapter): The backend-specific adapter.
        """
        self.adapter = adapter

    def get_clusters(self, model: Any) -> Dict[int, List[str]]:
        """Identifies pruning clusters for the given model.
        
        Args:
            model (Any): The model to analyze.
            
        Returns:
            Dict[int, List[str]]: A mapping of cluster IDs to lists of layer names.
        """
        graph = self.adapter.trace_graph(model)
        return graph.get("clusters", {})

    def get_topology_type(self, model: Any) -> str:
        """Determines the broad architectural class of the model.
        
        Args:
            model (Any): The model to classify.
            
        Returns:
            str: 'sequential', 'residual', or 'concatenative'.
        """
        return self.adapter.classify_architecture(model)
