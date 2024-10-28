"""
Handles stock clustering for model training.
Groups stocks based on multiple characteristics for efficient model training.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ClusterConfig:
    """Configuration for clustering parameters."""
    n_size_clusters: int = 3  # Large, Mid, Small Cap
    n_volatility_clusters: int = 3  # High, Med, Low Vol
    n_beta_clusters: int = 3  # High, Med, Low Beta
    min_cluster_size: int = 20
    max_cluster_size: int = 100

class UniverseClusterer:
    """Handles stock clustering based on multiple characteristics."""
    
    def __init__(self, config: ClusterConfig = None):
        """
        Initialize UniverseClusterer.
        
        Args:
            config: Clustering configuration
        """
        self.config = config or ClusterConfig()
        self.clusters: Dict[str, Dict[str, List[str]]] = {}
        self.cluster_stats: Dict[str, Dict[str, Any]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
    def create_clusters(self, 
                       stock_features: Dict[str, Dict],
                       sector_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Create hierarchical clusters within sectors.
        
        Args:
            stock_features: Dictionary of stock features
            sector_groups: Stocks grouped by sector
            
        Returns:
            Dictionary of cluster assignments
        """
        try:
            logger.info("Creating stock clusters...")
            
            for sector, symbols in sector_groups.items():
                if len(symbols) < self.config.min_cluster_size:
                    # Small sectors don't get sub-clustered
                    self.clusters[sector] = {'single': symbols}
                    continue
                    
                # Create feature matrix for clustering
                features_df = self._create_feature_matrix(symbols, stock_features)
                
                # Perform hierarchical clustering
                sector_clusters = self._cluster_sector(features_df)
                
                # Calculate cluster statistics
                self._calculate_cluster_stats(sector, sector_clusters, features_df)
                
                self.clusters[sector] = sector_clusters
                
            logger.info("Clustering complete")
            return self.clusters
            
        except Exception as e:
            logger.error(f"Error creating clusters: {e}")
            raise
            
    def _create_feature_matrix(self, 
                             symbols: List[str],
                             stock_features: Dict[str, Dict]) -> pd.DataFrame:
        """Create feature matrix for clustering."""
        features = []
        valid_symbols = []
        
        for symbol in symbols:
            if symbol not in stock_features:
                continue
                
            feature_dict = stock_features[symbol]
            features.append({
                'market_cap_log': np.log(feature_dict['market_cap'] + 1),
                'volatility': feature_dict['volatility'],
                'beta': feature_dict['beta'],
                'volume_log': np.log(feature_dict['avg_volume'] + 1)
            })
            valid_symbols.append(symbol)
            
        df = pd.DataFrame(features, index=valid_symbols)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        self.scalers[symbols[0][:3]] = scaler  # Store scaler using sector prefix
        
        return pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
        
    def _cluster_sector(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Perform hierarchical clustering within a sector."""
        clusters = {}
        
        # Size clustering
        size_clusters = self._perform_kmeans(
            features_df['market_cap_log'].values.reshape(-1, 1),
            self.config.n_size_clusters
        )
        
        # For each size cluster, create volatility clusters
        for size_idx in range(self.config.n_size_clusters):
            size_mask = size_clusters == size_idx
            size_group = features_df[size_mask]
            
            if len(size_group) < self.config.min_cluster_size:
                # Too small for sub-clustering
                clusters[f'size_{size_idx}'] = size_group.index.tolist()
                continue
                
            vol_clusters = self._perform_kmeans(
                size_group['volatility'].values.reshape(-1, 1),
                self.config.n_volatility_clusters
            )
            
            # For each volatility cluster, create beta clusters
            for vol_idx in range(self.config.n_volatility_clusters):
                vol_mask = vol_clusters == vol_idx
                vol_group = size_group[vol_mask]
                
                if len(vol_group) < self.config.min_cluster_size:
                    clusters[f'size_{size_idx}_vol_{vol_idx}'] = vol_group.index.tolist()
                    continue
                    
                beta_clusters = self._perform_kmeans(
                    vol_group['beta'].values.reshape(-1, 1),
                    self.config.n_beta_clusters
                )
                
                for beta_idx in range(self.config.n_beta_clusters):
                    beta_mask = beta_clusters == beta_idx
                    cluster_name = f'size_{size_idx}_vol_{vol_idx}_beta_{beta_idx}'
                    clusters[cluster_name] = vol_group[beta_mask].index.tolist()
                    
        return clusters
        
    def _perform_kmeans(self, 
                       data: np.ndarray, 
                       n_clusters: int,
                       random_state: int = 42) -> np.ndarray:
        """Perform K-means clustering."""
        if len(data) < n_clusters:
            return np.zeros(len(data))
            
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        return kmeans.fit_predict(data)
        
    def _calculate_cluster_stats(self,
                               sector: str,
                               clusters: Dict[str, List[str]],
                               features_df: pd.DataFrame) -> None:
        """Calculate statistics for each cluster."""
        stats = {}
        
        for cluster_name, symbols in clusters.items():
            cluster_features = features_df.loc[symbols]
            
            stats[cluster_name] = {
                'size': len(symbols),
                'avg_market_cap': np.exp(cluster_features['market_cap_log'].mean()) - 1,
                'avg_volatility': cluster_features['volatility'].mean(),
                'avg_beta': cluster_features['beta'].mean(),
                'avg_volume': np.exp(cluster_features['volume_log'].mean()) - 1,
                'homogeneity': self._calculate_homogeneity(cluster_features)
            }
            
        self.cluster_stats[sector] = stats
        
    def _calculate_homogeneity(self, features: pd.DataFrame) -> float:
        """Calculate cluster homogeneity score."""
        # Use average pairwise distance as homogeneity measure
        distances = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                distance = np.sqrt(np.sum((features.iloc[i] - features.iloc[j]) ** 2))
                distances.append(distance)
                
        return 1 / (1 + np.mean(distances)) if distances else 0
        
    def get_cluster_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all clusters."""
        return self.cluster_stats
        
    def save_clusters(self, filepath: str) -> None:
        """Save cluster assignments and statistics."""
        data = {
            'clusters': self.clusters,
            'stats': self.cluster_stats,
            'config': vars(self.config)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_clusters(self, filepath: str) -> None:
        """Load cluster assignments and statistics."""
        with open(filepath) as f:
            data = json.load(f)
            
        self.clusters = data['clusters']
        self.cluster_stats = data['stats']
        self.config = ClusterConfig(**data['config'])
        
    def get_symbols_in_cluster(self, 
                             sector: str, 
                             cluster_name: str) -> List[str]:
        """Get symbols in a specific cluster."""
        return self.clusters.get(sector, {}).get(cluster_name, [])
        
    def get_cluster_for_symbol(self, 
                             symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find cluster assignment for a symbol.
        
        Returns:
            Tuple of (sector, cluster_name) or (None, None) if not found
        """
        for sector, sector_clusters in self.clusters.items():
            for cluster_name, symbols in sector_clusters.items():
                if symbol in symbols:
                    return sector, cluster_name
        return None, None