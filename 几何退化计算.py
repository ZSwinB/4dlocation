import numpy as np
import itertools
import matplotlib.pyplot as plt

def analyze_geometric_degeneracy(receiver_positions, angle_threshold=15):
    """
    Given a list of 2D receiver positions (shape: Nx2), analyze all 3-combinations
    and compute their minimum angle and triangle area to assess geometric degeneracy.

    Parameters:
        receiver_positions (np.ndarray): Nx2 array of (x, y) receiver coordinates.
        angle_threshold (float): Threshold in degrees below which a combination is considered degenerate.

    Returns:
        sorted_results (list of dict): List containing combination index, angle, and area.
    """

    def angle_between(v1, v2):
        cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    results = []
    combinations = list(itertools.combinations(range(len(receiver_positions)), 3))

    for comb in combinations:
        p1, p2, p3 = receiver_positions[comb[0]], receiver_positions[comb[1]], receiver_positions[comb[2]]
        v1, v2, v3 = p2 - p1, p3 - p1, p3 - p2

        angle1 = angle_between(v1, v2)
        angle2 = angle_between(-v1, v3)
        angle3 = angle_between(-v2, -v3)
        min_angle = min(angle1, angle2, angle3)

        # Triangle area
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        s = (a + b + c) / 2
        try:
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        except:
            area = 0.0

        results.append({
            "combination": comb,
            "min_angle_deg": min_angle,
            "triangle_area": area,
            "is_degenerate": min_angle < angle_threshold
        })

    # Sort by angle (ascending = most degenerate first)
    sorted_results = sorted(results, key=lambda x: x["min_angle_deg"])

    # Plot
    min_angles = [r["min_angle_deg"] for r in sorted_results]
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(min_angles)), min_angles, c='red')
    plt.axhline(angle_threshold, linestyle='--', color='gray', label=f'Degeneracy Threshold ({angle_threshold}°)')
    plt.title("Minimum Angle of Receiver Triplets (Lower = More Degenerate)")
    plt.xlabel("Triplet Index (sorted)")
    plt.ylabel("Minimum Angle (degrees)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return sorted_results
