#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

# -------------------------------
# Configuration - adjust as needed
# -------------------------------
EXCEL_FILE = "./Tundra_Tongue_Data.xlsx"     # Your Excel file with columns as described
INPUT_FOLDER = "./main/raw_data/motor_and_loadcell"       # Folder containing CSV files
OUTPUT_FOLDER = "./output"
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

# We merge short gaps (< 0.1 s) between force-app zones and discard zones < 0.1 s.
MIN_GAP_DURATION = 0.1
MIN_ZONE_DURATION = 0.1

# ----------------------------------------------------------------
# 1) Force-app zone detection
# ----------------------------------------------------------------
def compute_force_app_zones(df, contact_start_idx, detach_idx, force_threshold):
    """
    Between contact_start_idx and detach_idx, identify zones where df["Value"] > force_threshold.
    Merge short gaps (< MIN_GAP_DURATION) and discard zones shorter than MIN_ZONE_DURATION.

    Returns:
      valid_zones: list of (start_time, end_time)
      total_time: total duration of all zones
    """
    df_zone = df.iloc[contact_start_idx:detach_idx+1].copy()
    time_vals = df_zone["Time_s"].values
    force_vals = df_zone["Value"].values

    # Identify when force is above the threshold
    active = force_vals > force_threshold

    raw_zones = []
    in_zone = False
    zone_start = None

    for i, is_active in enumerate(active):
        if is_active and not in_zone:
            in_zone = True
            zone_start = time_vals[i]
        elif not is_active and in_zone:
            # End of a zone
            zone_end = time_vals[i-1]
            raw_zones.append((zone_start, zone_end))
            in_zone = False
    # If still in a zone at the end
    if in_zone:
        raw_zones.append((zone_start, time_vals[-1]))

    # Merge short gaps
    merged_zones = []
    if raw_zones:
        current_start, current_end = raw_zones[0]
        for (start, end) in raw_zones[1:]:
            if start - current_end < MIN_GAP_DURATION:
                # merge
                current_end = end
            else:
                merged_zones.append((current_start, current_end))
                current_start, current_end = start, end
        merged_zones.append((current_start, current_end))

    # Filter out zones shorter than MIN_ZONE_DURATION
    valid_zones = []
    total_time = 0.0
    for (start, end) in merged_zones:
        duration = end - start
        if duration >= MIN_ZONE_DURATION:
            valid_zones.append((start, end))
            total_time += duration

    return valid_zones, total_time

# ----------------------------------------------------------------
# 2) Process a single CSV file using the weight as force threshold
# ----------------------------------------------------------------
def process_file(file_path, weight_threshold):
    """
    Reads CSV, sorts by timestamp, computes:
      - contact start (first force > 15g)
      - detach point (lowest force after contact)
      - contact duration
      - force application zones (threshold = weight_threshold)
    Returns (metrics, df).
    """
    df = pd.read_csv(file_path)
    df.sort_values("Timestamp(ns)", inplace=True, ignore_index=True)

    # Convert from ns to seconds
    t0 = df.loc[0, "Timestamp(ns)"]
    df["Time_s"] = (df["Timestamp(ns)"] - t0) / 1e9

    # Contact start: first force > 15g
    contact_indices = df.index[df["Value"] > 15].tolist()
    if not contact_indices:
        contact_start_idx = None
        contact_start_time = None
    else:
        contact_start_idx = contact_indices[0]
        contact_start_time = df.loc[contact_start_idx, "Time_s"]

    # Detach point: minimum force after contact
    if contact_start_idx is not None:
        contact_segment = df.iloc[contact_start_idx:]
        detach_idx = contact_segment["Value"].idxmin()
        detach_time = df.loc[detach_idx, "Time_s"]
        detach_force = df.loc[detach_idx, "Value"]
        contact_duration = detach_time - contact_start_time
    else:
        detach_idx = None
        detach_time = None
        detach_force = None
        contact_duration = None

    # Force threshold = weight_threshold from Excel
    force_threshold = weight_threshold

    # Force-application zones
    if contact_start_idx is not None and detach_idx is not None:
        zones, total_force_app_time = compute_force_app_zones(
            df, contact_start_idx, detach_idx, force_threshold
        )
        num_zones = len(zones)
    else:
        zones, total_force_app_time, num_zones = [], 0.0, 0

    metrics = {
        "contact_start_time": contact_start_time,
        "detach_time": detach_time,
        "contact_duration": contact_duration,
        "detach_force": detach_force,
        "force_app_zones": zones,
        "num_force_applications": num_zones,
        "total_force_app_time": total_force_app_time,
        "force_threshold": force_threshold
    }
    return metrics, df

# ----------------------------------------------------------------
# 3) Plotting
# ----------------------------------------------------------------
def plot_data(df, metrics, batch, tongue, placement, force_label, save_path):
    """
    Plots force vs time. Highlights:
      - contact start (green line)
      - detach (red line)
      - motor on (orange shading) if "MotorActive" toggles
      - force application zones (cyan shading above threshold)
    Title includes batch, tongue, weight, placement, and "force_label" (the 'Kraft' value).
    """
    time = df["Time_s"]
    force = df["Value"]
    weight = metrics["force_threshold"]

    # Make the plot a bit smaller
    plt.figure(figsize=(8, 4))
    plt.plot(time, force, label="Force (g)", color='blue')

    # Shade motor-active if column exists
    if "MotorActive" in df.columns and df["MotorActive"].nunique() > 1:
        motor_mask = df["MotorActive"].astype(bool)
        plt.fill_between(
            time,
            force.min(),
            force.max(),
            where=motor_mask,
            color='orange',
            alpha=0.2,
            label="Motor On"
        )

    # Mark contact start
    if metrics["contact_start_time"] is not None:
        plt.axvline(metrics["contact_start_time"], color='green', linestyle='--', label="Contact Start")
        plt.text(
            metrics["contact_start_time"],
            force.max() * 0.9,
            f"Contact Start\n({metrics['contact_start_time']:.2f}s)",
            color='green',
            rotation=90,
            va='top'
        )

    # Mark detach
    if metrics["detach_time"] is not None:
        plt.axvline(metrics["detach_time"], color='red', linestyle='--', label="Detach")
        plt.plot(metrics["detach_time"], metrics["detach_force"], 'ro')
        plt.text(
            metrics["detach_time"],
            metrics["detach_force"],
            f"Peak Force: {abs(metrics['detach_force']):.1f}g",
            color='red',
            va='bottom',
            ha='right'
        )

    # Shade each force-app zone
    for i, (start, end) in enumerate(metrics["force_app_zones"]):
        label = "Force App Zone" if i == 0 else None
        plt.axvspan(start, end, color='cyan', alpha=0.3, label=label)
        plt.text(
            (start + end) / 2,
            force.max() * 0.7,
            f"Zone {i+1}\n({end - start:.2f}s)",
            color='black',
            ha='center'
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Force (g)")
    plt.title(f"Batch {batch}, Tongue {tongue} (Weight: {weight} g), Placement: {placement}, Kraft: {force_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

# ----------------------------------------------------------------
# 4) PDF generation for plots with metrics
# ----------------------------------------------------------------
def fmt(val, precision=2):
    """Helper to safely format a numeric value or return 'N/A'."""
    if val is None:
        return "N/A"
    try:
        return f"{val:.{precision}f}"
    except:
        return str(val)

def generate_pdf(results, output_pdf_path):
    """
    Creates a PDF with up to two plots per page, each with its own computed metrics.
    We add an extra gap before the second plot so that the text for the first plot
    has more space.
    """
    pdf = FPDF()
    # Fix for missing attribute in some fpdf2 versions
    if not hasattr(pdf, 'unifontsubset'):
        pdf.unifontsubset = False
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_font("Helvetica", size=10)
    
    margin = 10
    page_width = pdf.w - 2 * margin

    # Scale factor for images (try 0.7, 0.6, etc. if you need more or less space)
    scale = 0.7
    image_w = page_width * scale
    image_h = image_w / 2

    # We'll center the image horizontally
    def center_x(img_width):
        return margin + (page_width - img_width) / 2

    text_height = 28
    # Additional vertical space before second plot
    plot_gap = 30  # in mm; increase if you need more space

    # We go through 'results' in steps of 2 so each page can show 2 plots.
    for i in range(0, len(results), 2):
        pdf.add_page()
        y_position = margin

        # ------------------ FIRST PLOT & METRICS ------------------
        res1 = results[i]
        # 1) Plot image
        if os.path.exists(res1["plot_path"]):
            pdf.image(res1["plot_path"], x=center_x(image_w), y=y_position, w=image_w, h=image_h)
        else:
            pdf.set_xy(margin, y_position)
            pdf.cell(page_width, image_h, "Image not found", border=1, ln=1, align="C")
        y_position += image_h

        # 2) Metrics for the first test (immediately below the first plot)
        metrics_text_1 = (
            f"Batch: {res1['Batch']} | Tunge nr: {res1['Tunge nr']} | Plassering: {res1['Plassering']} | "
            f"Kraft: {res1['Kraft']} | Filnavn: {res1['Filnavn']}\n"
            f"Vekt [gr]: {res1['Vekt [gr]']} | Start time: {res1['Start time']} | End time: {res1['End time']}\n"
            f"Tongue Temp (IR): {res1['Tongue Temp (IR)']} | Pipe temp sensor: {res1['Pipe temp sensor']} | "
            f"Pipe temp IR: {res1['Pipe temp IR']}\n"
            f"Frostskade: {res1['Frostskade']} | Avrivningsskade: {res1['Avrivningsskade']} | "
            f"Dårlig kvalitet: {res1['Dårlig kvalitet']}\n"
            f"Kommentar: {res1['Kommentar']}\n\n"
            f"--- Computed Metrics ---\n"
            f"Contact Duration (s): {fmt(res1['contact_time(s)'])} | "
            f"Total Force App Time (s): {fmt(res1['total_force_app_time(s)'])}\n"
            f"Num Force Applications: {res1['num_force_applications']} | "
            f"Detach Force (g): {fmt(res1['detach_force(g)'])}"
        )
        pdf.set_xy(margin, y_position)
        pdf.multi_cell(page_width, 5, metrics_text_1, border=0)
        y_position += text_height

        # ------------------ SECOND PLOT & METRICS (if available) ------------------
        if i + 1 < len(results):
            # Add an extra gap before the second plot
            y_position += plot_gap

            res2 = results[i+1]
            # 1) Second plot image
            if os.path.exists(res2["plot_path"]):
                pdf.image(res2["plot_path"], x=center_x(image_w), y=y_position, w=image_w, h=image_h)
            else:
                pdf.set_xy(margin, y_position)
                pdf.cell(page_width, image_h, "Image not found", border=1, ln=1, align="C")
            y_position += image_h

            # 2) Metrics for the second test (immediately below the second plot)
            metrics_text_2 = (
                f"Batch: {res2['Batch']} | Tunge nr: {res2['Tunge nr']} | Plassering: {res2['Plassering']} | "
                f"Kraft: {res2['Kraft']} | Filnavn: {res2['Filnavn']}\n"
                f"Vekt [gr]: {res2['Vekt [gr]']} | Start time: {res2['Start time']} | End time: {res2['End time']}\n"
                f"Tongue Temp (IR): {res2['Tongue Temp (IR)']} | Pipe temp sensor: {res2['Pipe temp sensor']} | "
                f"Pipe temp IR: {res2['Pipe temp IR']}\n"
                f"Frostskade: {res2['Frostskade']} | Avrivningsskade: {res2['Avrivningsskade']} | "
                f"Dårlig kvalitet: {res2['Dårlig kvalitet']}\n"
                f"Kommentar: {res2['Kommentar']}\n\n"
                f"--- Computed Metrics ---\n"
                f"Contact Duration (s): {fmt(res2['contact_time(s)'])} | "
                f"Total Force App Time (s): {fmt(res2['total_force_app_time(s)'])}\n"
                f"Num Force Applications: {res2['num_force_applications']} | "
                f"Detach Force (g): {fmt(res2['detach_force(g)'])}"
            )
            pdf.set_xy(margin, y_position)
            pdf.multi_cell(page_width, 5, metrics_text_2, border=0)
            y_position += text_height

    pdf.output(output_pdf_path)
    print(f"Saved PDF to {output_pdf_path}")

# ----------------------------------------------------------------
# 5) Main routine
# ----------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    try:
        meta_df = pd.read_excel(EXCEL_FILE)
    except Exception as e:
        print(f"Error reading {EXCEL_FILE}: {e}")
        return

    # 1) Sort by Batch, then by Tunge nr
    meta_df = meta_df.sort_values(by=["Batch", "Tunge nr"], ascending=True)

    results = []

    for idx, row in meta_df.iterrows():
        # Pull columns from your Excel. If any column name differs, adjust below.
        tongue_num = row["Tunge nr"]            
        placement = row["Plassering"]           
        force_label = row["Kraft"]              
        batch = row["Batch"]                    
        file_name = str(row["Filnavn"]).strip()  
        weight = row["Vekt [gr]"]               
        start_time = row["Start time"]
        end_time = row["End time"]

        # Additional columns:
        tongue_temp_ir = row.get("Tongue Temp (IR)", None)
        pipe_temp_sensor = row.get("Pipe temp sensor", None)
        pipe_temp_ir = row.get("Pipe temp IR", None)
        frostskade = row.get("Frostskade", "")
        avrivningsskade = row.get("Avrivningsskade", "")
        darlig_kvalitet = row.get("Dårlig kvalitet", "")
        kommentar = row.get("Kommentar", "")

        # Ensure we have .csv extension
        if not file_name.lower().endswith(".csv"):
            file_name += ".csv"

        file_path = os.path.join(INPUT_FOLDER, file_name)
        if not os.path.isfile(file_path):
            print(f"CSV file '{file_path}' not found. Skipping row {idx}.")
            continue

        # 2) Process CSV to get computed metrics
        metrics, df = process_file(file_path, weight_threshold=weight)

        # 3) Plot and save PNG image
        plot_filename = os.path.splitext(file_name)[0] + ".png"
        plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
        plot_data(df, metrics, batch, tongue_num, placement, force_label, plot_path)

        # 4) Build a dictionary of all info needed for CSV & PDF
        results.append({
            # Original Excel columns
            "Batch": batch,
            "Tunge nr": tongue_num,
            "Plassering": placement,
            "Kraft": force_label,
            "Filnavn": file_name,
            "Vekt [gr]": weight,
            "Start time": start_time,
            "End time": end_time,
            "Tongue Temp (IR)": tongue_temp_ir,
            "Pipe temp sensor": pipe_temp_sensor,
            "Pipe temp IR": pipe_temp_ir,
            "Frostskade": frostskade,
            "Avrivningsskade": avrivningsskade,
            "Dårlig kvalitet": darlig_kvalitet,
            "Kommentar": kommentar,
            # Computed metrics
            "contact_time(s)": metrics["contact_duration"],
            "total_force_app_time(s)": metrics["total_force_app_time"],
            "num_force_applications": metrics["num_force_applications"],
            "detach_force(g)": abs(metrics["detach_force"]) if metrics["detach_force"] is not None else None,
            # For PDF
            "plot_path": plot_path
        })

    # 5) Write summary CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(OUTPUT_FOLDER, "data_results_loadcellcalibrated.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved summary results to {results_csv}")

    # 6) Generate PDF with two tongue plots per page, with extra gap before second plot
    output_pdf_path = os.path.join(OUTPUT_FOLDER, "tongue_plots_loadcellcalibrated.pdf")
    generate_pdf(results, output_pdf_path)

if __name__ == "__main__":
    main()
