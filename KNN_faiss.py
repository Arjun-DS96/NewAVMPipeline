import numpy as np
import faiss
import pandas as pd
from typing import List, Tuple, Union, Optional
import math

class GeoKNNSearch:
    """
    A class for finding K-nearest neighbors based on latitude and longitude coordinates.
    Uses FAISS for efficient similarity search on large datasets.
    
    The input DataFrame should have columns for property index, latitude, and longitude.
    """
    
    def __init__(self, 
                 data: Optional[pd.DataFrame] = None,
                 lat_col: str = 'latitude', 
                 lon_col: str = 'longitude',
                 id_col: str = 'property_index',
                 use_exact_distance: bool = True):
        """
        Initialize the GeoKNNSearch class with an optional DataFrame.
        
        Args:
            data: DataFrame containing property data with latitude and longitude
            lat_col: Name of the latitude column in the DataFrame
            lon_col: Name of the longitude column in the DataFrame
            id_col: Name of the property index column in the DataFrame
            use_exact_distance: If True, uses Haversine distance for final ranking.
                               If False, uses Euclidean distance approximation.
        """
        self.index = None
        self.locations = None
        self.property_ids = None
        self.use_exact_distance = use_exact_distance
        self.earth_radius_km = 6371.0  # Earth radius in kilometers
        
        # If data is provided, fit immediately
        if data is not None:
            self.fit_from_dataframe(data, lat_col, lon_col, id_col)
        
    def fit_from_dataframe(self, 
                          data: pd.DataFrame, 
                          lat_col: str = 'latitude', 
                          lon_col: str = 'longitude',
                          id_col: str = 'property_index'):
        """
        Build the search index from a DataFrame containing latitude/longitude coordinates.
        
        Args:
            data: DataFrame containing property data
            lat_col: Name of the latitude column in the DataFrame
            lon_col: Name of the longitude column in the DataFrame
            id_col: Name of the property index column in the DataFrame
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
            
        # Verify columns exist
        for col in [lat_col, lon_col, id_col]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
        
        # Extract coordinates and property IDs
        coordinates = data[[lat_col, lon_col]].values
        self.property_ids = data[id_col].values
        
        # Fit the model
        self.fit(coordinates.tolist())  # Convert to list to avoid ambiguity
        
        return self
        
    def fit(self, coordinates: List[Tuple[float, float]]):
        """
        Build the search index from latitude/longitude coordinates.
        
        Args:
            coordinates: List of (latitude, longitude) tuples or numpy array
        """
        if len(coordinates) == 0:  # Check for empty list
            raise ValueError("Empty coordinates list provided")
            
        # Store original coordinates
        self.locations = np.array(coordinates, dtype=np.float32)
        
        # Convert geographic coordinates to feature vectors
        vectors = self._coordinates_to_vectors(self.locations)
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        
        return self
    
    def _coordinates_to_vectors(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Convert geographic coordinates to vectors suitable for FAISS.
        Simple conversion to approximate proximity.
        
        Args:
            coordinates: Array of [latitude, longitude] points
            
        Returns:
            Array of feature vectors
        """
        # Scale longitude differences by cosine of latitude to account for 
        # the convergence of meridians at higher latitudes
        lat_rad = np.radians(coordinates[:, 0])
        lon_scaled = coordinates[:, 1] * np.cos(lat_rad)
        
        # Return as 2D vectors
        return np.column_stack([coordinates[:, 0], lon_scaled]).astype(np.float32)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two points on Earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point (in degrees)
            lat2, lon2: Latitude and longitude of second point (in degrees)
            
        Returns:
            Distance in kilometers
        """
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Return distance in kilometers
        return self.earth_radius_km * c
    
    def knearest(self, 
                query: Tuple[float, float], 
                k: int = 5,
                return_distances: bool = True) -> Union[List, Tuple[List, List[float]]]:
        """
        Find k-nearest neighbors to the query point.
        
        Args:
            query: Query point as (latitude, longitude)
            k: Number of nearest neighbors to return
            return_distances: If True, also return distances
            
        Returns:
            If return_distances is False, returns property indices of nearest neighbors.
            If return_distances is True, returns (property_indices, distances) tuple.
        """
        if self.index is None:
            raise ValueError("Model not fitted. Call fit() or fit_from_dataframe() first.")
            
        # Convert query to vector
        query_vector = self._coordinates_to_vectors(np.array([query])).astype(np.float32)
        
        # FAISS search
        D, I = self.index.search(query_vector, k)
        
        # Convert FAISS indices to property indices
        if self.property_ids is not None:
            property_indices = [self.property_ids[idx] for idx in I[0]]
        else:
            property_indices = I[0].tolist()
        
        if self.use_exact_distance and return_distances:
            # Recalculate distances using Haversine formula for accuracy
            distances = []
            for idx in I[0]:
                lat, lon = self.locations[idx]
                dist = self._haversine_distance(query[0], query[1], lat, lon)
                distances.append(dist)
            
            # Return property indices and distances
            return property_indices, distances
        elif return_distances:
            # Return approximate distances from FAISS
            return property_indices, D[0].tolist()
        else:
            # Return only property indices
            return property_indices
            
    def radius_search(self, 
                      query: Tuple[float, float], 
                      radius_km: float, 
                      max_results: int = 1000) -> List:
        """
        Find all neighbors within a given radius.
        
        Args:
            query: Query point as (latitude, longitude)
            radius_km: Search radius in kilometers
            max_results: Maximum number of results to return
            
        Returns:
            Property indices of points within the radius
        """
        if self.index is None:
            raise ValueError("Model not fitted. Call fit() or fit_from_dataframe() first.")
            
        # First get more candidates than needed (approximate search)
        indices, distances = self.knearest(query, k=min(max_results, len(self.locations)), return_distances=True)
        
        # Filter by actual distance
        result_indices = []
        for idx, dist in zip(indices, distances):
            if dist <= radius_km:
                result_indices.append(idx)
                
        return result_indices