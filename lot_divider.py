from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define a rectangle as (x_min, x_max, y_min, y_max)
Rectangle = Tuple[float, float, float, float]

def area(rect: Rectangle) -> float:
    x_min, x_max, y_min, y_max = rect
    return (x_max - x_min) * (y_max - y_min)

def divide(rect: Rectangle, A_threshold: float) -> List[Rectangle]:
    if area(rect) <= A_threshold:
        return [rect]

    x_min, x_max, y_min, y_max = rect
    width = x_max - x_min
    height = y_max - y_min

    if height > width:
        y_mid = (y_min + y_max) / 2
        R1 = (x_min, x_max, y_min, y_mid)
        R2 = (x_min, x_max, y_mid, y_max)
    else:
        x_mid = (x_min + x_max) / 2
        R1 = (x_min, x_mid, y_min, y_max)
        R2 = (x_mid, x_max, y_min, y_max)

    return divide(R1, A_threshold) + divide(R2, A_threshold)

# Optional: control max building height based on area
def max_height(rect: Rectangle, k: float) -> float:
    return k * area(rect)

def visualize_lots(block: Rectangle, lots: List[Rectangle]):
    """Visualizes the lots using matplotlib."""
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', 'box')

    # Draw the main block
    x_min, x_max, y_min, y_max = block
    block_patch = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(block_patch)

    # Draw each lot
    for lot in lots:
        lx_min, lx_max, ly_min, ly_max = lot
        width = lx_max - lx_min
        height = ly_max - ly_min
        rect_patch = patches.Rectangle((lx_min, ly_min), width, height, linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax.add_patch(rect_patch)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Recursive Lot Division")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    # Save the plot to a file
    plt.savefig("lots.png")
    print("\nPlot saved to lots.png")


# Example usage
if __name__ == "__main__":
    block = (0.0, 100.0, 0.0, 80.0)
    A_threshold = 200.0
    k = 0.05  # Max height per unit area

    lots = divide(block, A_threshold)

    for i, lot in enumerate(lots):
        h_max = max_height(lot, k)
        print(f"Lot {i+1}: {lot}, Area = {area(lot):.2f}, Max Height = {h_max:.2f}")

    visualize_lots(block, lots)
