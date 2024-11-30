import pyvista as pv


def read_vtu(vtu_file):
    """Read VTU file and return mesh"""
    return pv.read(vtu_file)


def create_mesh_kwargs(field_name, cmap, vmin, vmax):
    """Create base mesh kwargs dictionary"""
    mesh_kwargs = {"scalars": field_name, "cmap": cmap}
    if vmin is not None:
        mesh_kwargs["clim"] = [vmin, vmax]
    return mesh_kwargs


def set_display_mode(mesh_kwargs, mode):
    """Update mesh kwargs based on display mode"""
    match mode:
        case "surface":
            mesh_kwargs["show_edges"] = False
        case "mesh":
            mesh_kwargs["show_edges"] = True
            mesh_kwargs["style"] = "surface"
        case "points":
            mesh_kwargs["style"] = "points"
            mesh_kwargs["point_size"] = 5
    return mesh_kwargs


def create_plotter(mesh, mesh_kwargs):
    """Create and initialize plotter"""
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, **mesh_kwargs)
    plotter.view_xy()
    return plotter


def setup_key_bindings(
    plotter, mesh, modes, current_mode, field_name, cmap, vmin, vmax
):
    """Setup key bindings for view control"""

    def update_display(mode):
        mesh_kwargs = create_mesh_kwargs(field_name, cmap, vmin, vmax)
        mesh_kwargs = set_display_mode(mesh_kwargs, mode)
        plotter.clear()
        plotter.add_mesh(mesh, **mesh_kwargs)

    def cycle_display_mode(event=None):
        current_idx = modes.index(current_mode["mode"])
        next_idx = (current_idx + 1) % len(modes)
        current_mode["mode"] = modes[next_idx]
        update_display(current_mode["mode"])

    def reset_view(event=None):
        plotter.reset_camera()
        plotter.view_xy()

    plotter.add_key_event("r", reset_view)
    plotter.add_key_event("m", cycle_display_mode)


def visualize(
    mesh, field_name="p", cmap="coolwarm", vmin=None, vmax=None, display_mode="mesh"
):
    modes = ["mesh", "points", "surface"]
    current_mode = {"mode": display_mode}

    mesh_kwargs = create_mesh_kwargs(field_name, cmap, vmin, vmax)
    mesh_kwargs = set_display_mode(mesh_kwargs, display_mode)
    plotter = create_plotter(mesh, mesh_kwargs)

    # Pass all required parameters
    setup_key_bindings(plotter, mesh, modes, current_mode, field_name, cmap, vmin, vmax)

    plotter.show()


# Usage example
if __name__ == "__main__":
    dataset_path = "/mnt/c/Users/victo/Downloads/Dataset/Dataset"
    case_name = "airFoil2D_SST_47.017_-4.369_4.85_6.296_6.401"
    suffix = "_internal.vtu"
    vtu_file = "/".join([dataset_path, case_name, case_name + suffix])
    mesh = read_vtu(vtu_file)
    visualize(mesh, "U", "viridis", vmin=0, vmax=70, display_mode="surface")
