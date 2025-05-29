from build123d import *
from ocp_vscode import show_object
import numpy as np
import math

# Toggle DXF export on/off
EXPORT_DXF = False

# Helper for DXF exporting
from build123d import Unit

def export_dxf(shape, layer: str, filename: str):
    if EXPORT_DXF:
        exporter = ExportDXF(unit=Unit.MM)
        exporter.add_shape(shape, layer=layer)
        exporter.write(filename)

# For reproducibility (optional)
numpy_random = np.random.seed(42)

# ---------------------------------------------------------
# Parameters: overall width and layer heights (bottom-up)
# ---------------------------------------------------------
width = 10  # full horizontal extent

# Only these layers remain as continuous regions:
# Fat (bottom), Connective Tissue, Epitell, Keratin, Saliva.
fat_nominal_height    = 6.76        # nominal height for fat region
connective_tissue_height = 0.824    # nominal vertical extent of connective tissue
epitell_height        = 0.127
keratin_height        = 0.168
saliva_height         = 0.01
metal_plate_height    = 4.6

overall_length = (
    fat_nominal_height
    + connective_tissue_height
    + epitell_height
    + keratin_height
    + saliva_height
)

# ---------------------------------------------------------
# Functions for variable amplitude and frequency
# ---------------------------------------------------------
def fat_ct_amplitude(x):
    base_amplitude = 0.1 + (0.2 - 0.1) * (x / width)
    return base_amplitude + np.random.uniform(-0.02, 0.02)

def fat_ct_frequency(x):
    base_frequency = 5.0 + (3.0 - 5.0) * (x / width)
    return base_frequency + np.random.uniform(-0.5, 0.5)

def ct_epitell_amplitude(x):
    base_amplitude = 0.08 + (0.12 - 0.08) * (x / width)
    return base_amplitude + np.random.uniform(-0.01, 0.01)

def ct_epitell_frequency(x):
    base_frequency = 38.0 + (42.0 - 38.0) * (x / width)
    return base_frequency + np.random.uniform(-2.0, 2.0)

# ---------------------------------------------------------
# Sinus Interface Points
# ---------------------------------------------------------
n_samples_fat_ct = 200
x_samples_fat_ct = np.linspace(0, width, n_samples_fat_ct)
fat_ct_interface_points = [
    (
        x,
        fat_nominal_height + fat_ct_amplitude(x) * math.sin(
            2 * math.pi * fat_ct_frequency(x) * (x / width)
        ),
    )
    for x in x_samples_fat_ct
]

n_samples_ct_epitell = 200
nominal_ct_epitell = fat_nominal_height + connective_tissue_height
x_samples_ct_epitell = np.linspace(0, width, n_samples_ct_epitell)
ct_epitell_interface_points = [
    (
        x,
        nominal_ct_epitell + ct_epitell_amplitude(x) * math.sin(
            2 * math.pi * ct_epitell_frequency(x) * (x / width)
        ),
    )
    for x in x_samples_ct_epitell
]

avg_ct_epitell_y = np.mean([pt[1] for pt in ct_epitell_interface_points])
epitell_top = avg_ct_epitell_y + epitell_height

# ---------------------------------------------------------
# Define Colors for Each Region (for display purposes)
# ---------------------------------------------------------
fat_color         = (0.8, 0.8, 0.8)     # light gray
connective_color  = (1.0, 1.0, 0.0)     # yellow
epitell_color     = (0.0, 0.0, 1.0)     # blue
keratin_color     = (0.0, 1.0, 0.0)     # green
saliva_color      = (0.0, 1.0, 1.0)     # cyan
metal_plate_color = (0.5, 0.5, 0.5)     # gray

# ---------------------------------------------------------
# Build and Export Shapes
# ---------------------------------------------------------
# Overall outline
with BuildPart() as base_part:
    with BuildSketch(Plane.XY) as overall_sketch:
        with BuildLine():
            Line((0, 0), (width, 0))
            Line((width, 0), (width, overall_length))
            Line((width, overall_length), (0, overall_length))
            Line((0, overall_length), (0, 0))
        overall_face = make_face()
    show_object(overall_face)
export_dxf(overall_face, "Overall", "./ParameterizedDesign/DXF/overall_face.dxf")

# Fat region
with BuildSketch(Plane.XY) as fat_sketch:
    with BuildLine():
        Line((0, 0), (width, 0))
        Line((width, 0), (width, fat_ct_interface_points[-1][1]))
        Spline(fat_ct_interface_points)
        Line((0, fat_ct_interface_points[0][1]), (0, 0))
    fat_face = make_face()
show_object(fat_face, options={"color": fat_color})
export_dxf(fat_face, "Fat", "ParameterizedDesign/DXF/fat_face.dxf")

# Connective tissue (outer)
with BuildSketch(Plane.XY) as connective_sketch:
    with BuildLine():
        Spline(fat_ct_interface_points)
        Line((width, fat_ct_interface_points[-1][1]), (width, ct_epitell_interface_points[-1][1]))
        Spline(ct_epitell_interface_points)
        Line((0, ct_epitell_interface_points[0][1]), (0, fat_ct_interface_points[0][1]))
    outer_connective_face = make_face()
show_object(outer_connective_face, options={"color": connective_color})

# Create and cut circles in connective tissue
n_circles = 20
mean_area_target = 0.025
sigma_area = 0.025
circle_areas = np.clip(
    np.random.normal(loc=mean_area_target, scale=sigma_area, size=n_circles),
    a_min=mean_area_target,
    a_max=None,
)
circle_radii = [math.sqrt(a / math.pi) for a in circle_areas]
conn_bottom = np.mean([pt[1] for pt in fat_ct_interface_points])
conn_top = np.mean([pt[1] for pt in ct_epitell_interface_points])

circle_centers = []
for i, r_new in enumerate(circle_radii):
    for _ in range(1000):
        x, y = np.random.uniform(r_new, width - r_new), np.random.uniform(conn_bottom + r_new, conn_top - r_new)
        if all(
            math.hypot(x - cx, y - cy) >= (r_new + cr)
            for (cx, cy), cr in zip(circle_centers, circle_radii[: len(circle_centers)])
        ):
            circle_centers.append((x, y))
            break

with BuildSketch(Plane.XY) as circles_sketch:
    for center, radius in zip(circle_centers, circle_radii):
        with Locations(center):
            Circle(radius)
    circles = circles_sketch.faces()
show_object(circles, options={"color": (1.0, 0.0, 0.0)})
export_dxf(circles, "Connective Tissue Circles", "ParameterizedDesign/DXF/connective_tissue_circles.dxf")
connective_face = outer_connective_face.cut(*circles)
show_object(connective_face, options={"color": connective_color})
export_dxf(connective_face, "Connective Tissue", "ParameterizedDesign/DXF/connective_face.dxf")

# Fat circles
n_fat_circles = 50
mean_area_fat = 0.05
sigma_area_fat = 0.05
fat_circle_areas = np.clip(
    np.random.normal(loc=mean_area_fat, scale=sigma_area_fat, size=n_fat_circles),
    a_min=mean_area_fat,
    a_max=None,
)
fat_circle_radii = [math.sqrt(a / math.pi) for a in fat_circle_areas]
fat_top_bound = min(pt[1] for pt in fat_ct_interface_points)

fat_circle_centers = []
for i, r_new in enumerate(fat_circle_radii):
    for _ in range(1000):
        x, y = np.random.uniform(r_new, width - r_new), np.random.uniform(r_new, fat_top_bound - r_new)
        if all(
            math.hypot(x - cx, y - cy) >= (r_new + cr)
            for (cx, cy), cr in zip(fat_circle_centers, fat_circle_radii[: len(fat_circle_centers)])
        ):
            fat_circle_centers.append((x, y))
            break

with BuildSketch(Plane.XY) as fat_circles_sketch:
    for center, radius in zip(fat_circle_centers, fat_circle_radii):
        with Locations(center):
            Circle(radius)
    fat_circles = fat_circles_sketch.faces()
show_object(fat_circles, options={"color": (1.0, 0.0, 1.0)})
export_dxf(fat_circles, "Fat Circles", "ParameterizedDesign/DXF/fat_circles.dxf")
fat_face_with_cuts = fat_face.cut(*fat_circles)
show_object(fat_face_with_cuts, options={"color": fat_color})
export_dxf(fat_face_with_cuts, "Fat", "ParameterizedDesign/DXF/fat_face_with_circles.dxf")

# Epitell region
with BuildSketch(Plane.XY) as epitell_sketch:
    with BuildLine():
        Spline(ct_epitell_interface_points)
        Line((width, ct_epitell_interface_points[-1][1]), (width, epitell_top))
        Line((width, epitell_top), (0, epitell_top))
        Line((0, epitell_top), (0, ct_epitell_interface_points[0][1]))
    epitell_face = make_face()
show_object(epitell_face, options={"color": epitell_color})
export_dxf(epitell_face, "Epitell", "ParameterizedDesign/DXF/epitell_face.dxf")

# Keratin region
with BuildSketch(Plane.XY) as keratin_sketch:
    with BuildLine():
        Line((0, epitell_top), (width, epitell_top))
        Line((width, epitell_top), (width, epitell_top + keratin_height))
        Line((width, epitell_top + keratin_height), (0, epitell_top + keratin_height))
        Line((0, epitell_top + keratin_height), (0, epitell_top))
    keratin_face = make_face()
show_object(keratin_face, options={"color": keratin_color})
export_dxf(keratin_face, "Keratin", "ParameterizedDesign/DXF/keratin_face.dxf")

# Saliva region
current_y = epitell_top + keratin_height
with BuildSketch(Plane.XY) as saliva_sketch:
    with BuildLine():
        Line((0, current_y), (width, current_y))
        Line((width, current_y), (width, current_y + saliva_height))
        Line((width, current_y + saliva_height), (0, current_y + saliva_height))
        Line((0, current_y + saliva_height), (0, current_y))
    saliva_face = make_face()
show_object(saliva_face, options={"color": saliva_color})
export_dxf(saliva_face, "Saliva", "ParameterizedDesign/DXF/saliva_face.dxf")

# Metal plate region
current_y += saliva_height
with BuildSketch(Plane.XY) as metal_plate_sketch:
    with BuildLine():
        Line((0, current_y), (width, current_y))
        Line((width, current_y), (width, current_y + metal_plate_height))
        Line((width, current_y + metal_plate_height), (0, current_y + metal_plate_height))
        Line((0, current_y + metal_plate_height), (0, current_y))
    metal_plate_face = make_face()
show_object(metal_plate_face, options={"color": metal_plate_color})
export_dxf(metal_plate_face, "Metal plate", "ParameterizedDesign/DXF/metal_plate_face.dxf")
