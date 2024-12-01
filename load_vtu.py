import pyvista as pv
import numpy as np
import torch


def load_vtu_data(filename):
    """Load VTU file and convert to tensor format with coordinates and field data"""
    # Read the VTU file
    mesh = pv.read(filename)

    # Get coordinates (x,y)
    points = mesh.points[:, :2]

    # Get all point data arrays
    features = []
    feature_names = []

    # Extract each field
    for name, data in mesh.point_data.items():
        if len(data.shape) > 1:  # Vector field
            for i in range(data.shape[1]):
                features.append(data[:, i])
                feature_names.append(f"{name}_{i}")
        else:  # Scalar field
            features.append(data)
            feature_names.append(name)

    # Stack features and combine with coordinates
    features = np.stack(features, axis=1)
    data_array = np.hstack([points, features])

    # Convert to torch tensor
    data_tensor = torch.FloatTensor(data_array)

    return data_tensor, feature_names


# Example usage:
if __name__ == "__main__":
    dataset_path = "/mnt/c/Users/victo/Downloads/Dataset/Dataset"
    case_name = "airFoil2D_SST_47.017_-4.369_4.85_6.296_6.401"
    suffix = "_internal.vtu"
    vtu_file = "/".join([dataset_path, case_name, case_name + suffix])
    data_tensor, feature_names = load_vtu_data(vtu_file)
    print("Feature names:", feature_names)
    print("Data shape:", data_tensor.shape)
    print("First 5 rows of data:")
    print(data_tensor[:5])
