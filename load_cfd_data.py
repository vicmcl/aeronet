import numpy as np
import torch
import pyvista as pv
from sklearn.neighbors import KDTree
from typing import Tuple, Optional, List


class CFDDataLoader:
    def __init__(
        self,
        vtu_file: str,
        vtp_file: Optional[str] = None,
        tolerance: float = 1e-10,
        drop: Optional[List[str]] = None,
    ):
        """Initialize CFD data loader."""
        self.vtu_file = vtu_file
        self.vtp_file = vtp_file
        self.tolerance = tolerance
        self.features_to_drop = drop or []

    def _load_fluid_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load fluid mesh data from VTU file."""
        mesh = pv.read(self.vtu_file)
        fluid_points = mesh.points[:, :2]

        features = []
        feature_names = []

        for name, data in mesh.point_data.items():
            if name in self.features_to_drop:
                print(f"Skipping feature: {name}")
                continue
            else:
                print(f"Loading feature: {name}")

            if len(data.shape) > 1:  # Vector field
                for i in range(data.shape[1]):
                    if f"{name}_{i}" in self.features_to_drop:
                        print(f"Skipping feature: {name}_{i}")
                        continue
                    else:
                        features.append(data[:, i])
                        feature_names.append(f"{name}_{i}")

            else:  # Scalar field
                features.append(data)
                feature_names.append(name)

        features = np.stack(features, axis=1)
        return fluid_points, features, feature_names

    def _load_surface_data(self) -> Optional[np.ndarray]:
        """Load surface mesh points from VTP file."""
        if self.vtp_file is None:
            return None
        surface_mesh = pv.read(self.vtp_file)
        return surface_mesh.points[:, :2]

    def _combine_points_and_features(
        self,
        fluid_points: np.ndarray,
        fluid_features: np.ndarray,
        surface_points: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine fluid and surface points with features."""
        point_type = np.zeros(len(fluid_points))

        if surface_points is not None:
            surface_features = np.zeros((len(surface_points), fluid_features.shape[1]))
            points = np.vstack([fluid_points, surface_points])
            features = np.vstack([fluid_features, surface_features])
            surface_indicators = np.ones(len(surface_points))
            point_type = np.hstack([point_type, surface_indicators])
        else:
            points = fluid_points
            features = fluid_features

        return points, features, point_type

    def _remove_duplicates(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """Remove duplicate fluid points near surface."""
        surface_mask = data_tensor[:, -1] == 1
        fluid_mask = ~surface_mask

        surface_coords = data_tensor[surface_mask][:, :2]
        fluid_coords = data_tensor[fluid_mask][:, :2]

        surface_tree = KDTree(surface_coords)
        distances, _ = surface_tree.query(fluid_coords, k=1)

        unique_points_mask = distances.flatten() >= self.tolerance
        filtered_fluid = data_tensor[fluid_mask][unique_points_mask]
        surface_points = data_tensor[surface_mask]

        print(f"{np.sum(~unique_points_mask)} surface points detected.")
        return torch.cat([filtered_fluid, surface_points], dim=0)

    def load_data(self) -> Tuple[torch.Tensor, List[str]]:
        """Load and process all CFD data."""
        # Load data
        fluid_points, fluid_features, feature_names = self.load_fluid_data()
        surface_points = self.load_surface_data()

        # Combine points and features
        points, features, point_type = self._combine_points_and_features(
            fluid_points, fluid_features, surface_points
        )

        # Add surface indicator to features
        features = np.column_stack([features, point_type])
        feature_names.append("is_surface")

        # Create tensor
        data_array = np.hstack([points, features])
        data_tensor = torch.FloatTensor(data_array)

        # Remove duplicates and return
        clean_data = self._remove_duplicates(data_tensor)
        return clean_data, feature_names


def sample_data(
    data_tensor: torch.Tensor, num_fluid: int = 1000, num_surface: Optional[int] = None
) -> torch.Tensor:
    """Sample points from point cloud data, separating fluid and surface points."""
    sample_data = data_tensor[
        np.random.choice(np.where(data_tensor[:, -1] == 0)[0], num_fluid, replace=False)
    ]

    if num_surface is not None:
        surface_data = data_tensor[
            np.random.choice(
                np.where(data_tensor[:, -1] == 1)[0], num_surface, replace=False
            )
        ]
        return torch.cat([sample_data, surface_data], dim=0)

    return sample_data


# Example usage:
if __name__ == "__main__":
    dataset_path = "/mnt/c/Users/victo/Downloads/Dataset/Dataset"
    case_name = "airFoil2D_SST_47.017_-4.369_4.85_6.296_6.401"
    vtu_file = f"{dataset_path}/{case_name}/{case_name}_internal.vtu"
    vtp_file = f"{dataset_path}/{case_name}/{case_name}_aerofoil.vtp"

    loader = CFDDataLoader(
        vtu_file, vtp_file, drop=["U_2", "implicit_distance", "vtkOriginalPointIds"]
    )
    data_tensor, feature_names = loader.load_data()
    sampled_data = sample_data(data_tensor, num_fluid=1000, num_surface=100)
    print("Features:", feature_names)
    print("Data shape:", data_tensor.shape)

    # Visualize sampled data
    # import matplotlib.pyplot as plt

    # plt.scatter(sampled_data[:, 0], sampled_data[:, 1], c=sampled_data[:, -1], s=5)
    # plt.show()
