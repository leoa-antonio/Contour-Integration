import numpy as np
from matplotlib.pyplot import subplots

# Grid Utility Functions
def get_grid_cell(position, image_size=512, grid_size=16):
    """
    Converts coordinate to its corresponding grid cell

    Parameters
    ----------
    position: array-like of shape (2, )
        (x, y) coordinate relative to the image center.
    image_size: int, optional
        Image width and height in pixels. Default is 512.
    grid_size: int, optional
        Grid_size in pixels. Default is 16.
    
    Returns
    -------
    grid_cell: array-like of shape (2, )
        Integer grid indices corresponding to "position".
    """
    position = np.array(position)
    cell_size = image_size / grid_size

    # Convert from image-centered coordinates to coordinates with
    # origin at the upper-left corner
    shifted_position = position + (image_size // 2)

    # Floor converts each pixel coordinate to its assigned cell
    grid_cell = np.floor(shifted_position / cell_size)

    return grid_cell.astype(int)

def is_valid_grid_cell(grid_cell, grid_size=16):
    return (0 <= grid_cell[0] < grid_size and 0 <= grid_cell[1] < grid_size)

# Contour Utility Function
def generate_start_position(rng, beta, distance_from_center=64):
    """
    Generate the starting position and initial direction of a contour path.

    The starting position is located at a fixed distance from the image
    center. The initial path direction is directed toward the image center
    with an angular deviation of either +beta or -beta.

    Parameters
    ----------
    beta : float
        Magnitude of the angular deviation from the direction toward the
        image center, in radians.
    distance_from_center : float, optional
        Distance of the starting position from the image center, in pixels.
        Default is 64.

    Returns
    -------
    start_position : tuple of float
        Image-centered (x, y) coordinates of the starting position.
    theta0 : float
        Initial direction of the contour path, in radians.
    """

    # Random position on circle
    angle = rng.uniform(0, 2 * np.pi)

    x = distance_from_center * np.cos(angle)
    y = distance_from_center * np.sin(angle)

    start_position = (x, y)

    # Direction from starting position toward image center
    theta_center = np.arctan2(-y, -x)

    # Initial direction is ±beta from centerward direction
    sign = rng.choice([-1, 1])
    theta0 = theta_center + sign * beta

    return start_position, theta0

# Contour Generation
def generate_vertices(rng, beta, step_length, start_vertex, theta0, 
                      jitter_max=np.deg2rad(10), n_contour_elements=12, grid_size=16):
    """
    Generate the vertices of a contour path following the stimulus-generation
    procedure of Field, Hayes, and Hess (1993).

    The path is constructed iteratively from an initial position and direction.
    Each successive segment changes direction by either +beta or -beta with
    additional uniform angular jitter. Contour Gabor elements are positioned
    at the midpoints of consecutive vertices. Each contour element must occupy
    a unique grid cell. If a proposed element falls in an occupied cell, the
    corresponding path segment is extended by D/4 until an unoccupied cell is
    reached. Paths that leave the stimulus grid are rejected and regenerated.

    Parameters
    ----------
    beta: float
        Curvature parameter in radians specifying the angular deviation between vertices.
    step_length: float
        Distance between each vertice in pixels.
    start_vertex: array-like of shape (2, )
        Start position for contour line defined by generate_start_position.
    theta0: float
        Starting angle in radians for the first vertex.
    jitter_max: float
        Upper bound of error term for updating theta, in radians.
    n_contour_elements: int
        Number of contour elements for the vertices to hold. Equal to n_vertices - 1.
    grid_size: int, optional
        Grid size in pixels. Default is 16
    
    Returns
    -------
    vertices: array-like of shape (n_contour_elements, )
        Contour vertices used to calculate contour patch positions
    thetas: array-like of shape (n_contour_elements, )
        Turning angle used for each contour step
    occupied_cells: set of tuple(int, int)
        Grid cells occupied by contour Gabor elements, represented as (col, row) index pairs.
    """
    # Continue until a valid contour line is generated
    while True:

        # Reset outputs for each attempt
        vertices = [start_vertex]
        thetas = []
        occupied_cells = set()

        prev_vertex = start_vertex
        theta = theta0

        # Assume path is valid unless boundary is crossed
        valid_path = True

        for step in range(n_contour_elements):

            thetas.append(theta)

            next_vertex = (
                prev_vertex[0] + step_length * np.cos(theta),
                prev_vertex[1] + step_length * np.sin(theta)
            )

            proposed_gabor = (
                (prev_vertex[0] + next_vertex[0]) / 2,
                (prev_vertex[1] + next_vertex[1]) / 2
            )

            grid_cell = tuple(get_grid_cell(proposed_gabor))

            # Check boundary
            if not is_valid_grid_cell(grid_cell, grid_size):
                valid_path = False
                break

            # Deal with an occupied cell
            while grid_cell in occupied_cells:

                next_vertex = (
                    next_vertex[0] + (step_length / 4) * np.cos(theta),
                    next_vertex[1] + (step_length / 4) * np.sin(theta)
                )

                proposed_gabor = (
                    (prev_vertex[0] + next_vertex[0]) / 2,
                    (prev_vertex[1] + next_vertex[1]) / 2
                )

                grid_cell = tuple(get_grid_cell(proposed_gabor))

                # D/4 extension itself could leave the image
                if not is_valid_grid_cell(grid_cell, grid_size):
                    valid_path = False
                    break

            # Exit the for-loop and regenerate entire path
            if not valid_path:
                break

            occupied_cells.add(grid_cell)

            prev_vertex = next_vertex
            vertices.append(next_vertex)

            # Generate direction for next segment
            sign = rng.choice([-1, 1])
            jitter = rng.uniform(-jitter_max, jitter_max)
            theta += sign * beta + jitter

        # Only return if all n_contour_patches were successfully generated
        if valid_path:
            return np.array(vertices), np.array(thetas), occupied_cells

def calculate_contour_positions(vertices):
    """Calculate Gabor positions as the midpoint of each successive vertex."""
    gabor_positions = (vertices[:-1] + vertices[1:]) / 2
    return gabor_positions

# Background Generation
def generate_background_elements(rng, occupied_cells, image_size=512, grid_size=16):
    """
    Randomly generate background positions and angles for gabor patches 
    from a uniform distribution. 

    Parameters
    ----------
    occupied_cells: set of tuple(int, int)
        Grid cells occupied by contour Gabor elements, represented as (col, row) index pairs.   
    image_size: int, optional
        Image width and height in pixels. Default is 512.
    grid_size: int, optional
        Grid_size in pixels. Default is 16.

    Returns
    ------
    background_positions: np.ndarray of shape (n_background_elements, 2)
        Image centered (x, y) positions of Gabor elements. 
    background_angles: np.ndarray of shape (n_background_elements, )
        Random orientations of the background Gabor elements, in radians.
    
    Notes
    -----
    The number of background elements is given by
    ``grid_size**2 - len(occupied_cells)``.
    """
    # Declare Output Arrays
    background_positions = []
    background_angles = []

    cell_size = image_size / grid_size

    # Loop through Grid Cells
    for col in range(grid_size):
        for row in range(grid_size):
            grid_cell = (col, row)

            # Skip Occupied Cells
            if grid_cell in occupied_cells:
                continue

            # Get Boundaries of Current Cell
            x_min = col * cell_size - image_size / 2
            x_max = x_min + cell_size
            
            y_min = row * cell_size - image_size / 2
            y_max = y_min + cell_size

            # Get Random Position within Cell
            x_pos, y_pos = rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)

            # Get Random Angle
            angle = rng.uniform(0, np.pi)

            # Append Background position and angles
            background_positions.append((x_pos, y_pos))
            background_angles.append(angle)

    return np.array(background_positions), np.array(background_angles)          

# Gabor Patch Generation
def generate_gabor_patch(x, y, orientation, period=8, phase=0, sigma=4, image_size=512):
    """
    Generate a Gabor patch following Field, Hayes, and Hess (1993). The equation is 
    G(x, y) = exp(-(-x^2 - y^2)/2*size^2) * cos(2*pi*(x*cos(theta) + y*sin(theta))/period + phase)

    Parameters
    ----------
    period : float
        Period p of the sinusoidal carrier, in pixels.
    phase : float
        Phase phi of the sinusoidal carrier, in radians.
    sigma : float
        Standard deviation of the Gaussian envelope, in pixels.
    orientation : float
        Visible orientation of the Gabor element, in radians.
    x, y : float
        Center of the Gabor in image-centered coordinates.
    image_size : int, optional
        Width and height of the image in pixels. Default is 512.

    Returns
    -------
    gabor : np.ndarray of shape (image_size, image_size)
        Generated Gabor patch.
    """

    # Generate image grid
    x_axis = np.arange(-image_size // 2, image_size // 2)
    y_axis = np.arange(-image_size // 2, image_size // 2)
    X, Y = np.meshgrid(x_axis, y_axis)

    # Center coordinates on Gabor
    X_c = X - x
    Y_c = Y - y

    # Carrier direction is perpendicular to visible Gabor orientation
    carrier_angle = orientation + np.pi / 2

    # Gaussian envelope
    gaussian = np.exp(-(X_c**2 + Y_c**2) / (2 * sigma**2))

    # Sinusoidal carrier
    carrier_position = (np.cos(carrier_angle) * X_c + np.sin(carrier_angle) * Y_c)

    sinusoid = np.cos(2 * np.pi * carrier_position / period + phase)

    return gaussian * sinusoid

# Experiment 4 Stimuli Generators
def generate_stimulus(rng, beta, step_length, contour_present=True, n_contour_elements=12, image_size=512):
    stimulus = np.zeros((image_size, image_size))

    start_position, theta0 = generate_start_position(rng, beta)
    vertices, contour_angles, occupied_cells = generate_vertices(rng, beta, step_length, start_position, theta0, n_contour_elements=n_contour_elements)

    # Positive Stimulus Generation
    if contour_present:
        contour_positions = calculate_contour_positions(vertices)
        background_positions, background_angles = generate_background_elements(rng, occupied_cells)

        # Add Contour Patches
        for i in range(n_contour_elements):
            x_position, y_position = contour_positions[i]
            patch = generate_gabor_patch(x_position, y_position, contour_angles[i])
            stimulus += patch

        # Add Background Patches
        for i in range(len(background_positions)):
            x_position, y_position = background_positions[i]
            patch = generate_gabor_patch(x_position, y_position, background_angles[i])
            stimulus += patch

    # Negative Stimulus Generation
    else:
        occupied_cells = set()
        background_positions, background_angles = generate_background_elements(rng, occupied_cells)

        # Add Patches
        for i in range(len(background_positions)):
            x_position, y_position = background_positions[i]
            patch = generate_gabor_patch(x_position, y_position, background_angles[i])
            stimulus += patch

    return stimulus

# Plot Example 
def plot_example_stimuli(positive, negative, figsize=(10, 10)):
    fig, ax = subplots(ncols=2, nrows=1, figsize=(10, 10))
    ax[0].imshow(positive, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Stimulus with Contour')

    ax[1].imshow(negative, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Stimulus without Contour')


