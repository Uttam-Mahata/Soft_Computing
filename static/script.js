const grid = document.getElementById("grid");
const labelInput = document.getElementById("label");

// Initialize a 5x5 matrix with 0s
let matrix = Array(5).fill().map(() => Array(5).fill(0));

// Track mouse press state
let isMouseDown = false;

// Create 5x5 grid cells and attach mouse events
for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");

        // Toggle cell on single click
        cell.addEventListener("click", () => toggleCell(row, col, cell));

        // Toggle cell while dragging if mouse is down
        cell.addEventListener("mouseover", () => {
            if (isMouseDown) {
                toggleCell(row, col, cell);
            }
        });

        grid.appendChild(cell);
    }
}

// Listen for mouse down and up events
grid.addEventListener("mousedown", () => isMouseDown = true);
grid.addEventListener("mouseup", () => isMouseDown = false);
document.body.addEventListener("mouseup", () => isMouseDown = false); // Handle mouse up outside the grid

function toggleCell(row, col, cellElement) {
    // Toggle between 1 and 0 in the matrix
    matrix[row][col] = 1 - matrix[row][col];

    // Toggle the 'filled' class for visual effect
    cellElement.classList.toggle("filled", matrix[row][col] === 1);
}

function saveData() {
    const label = labelInput.value.toUpperCase();
    if (!label || label.length !== 1 || !/[A-Z]/.test(label)) {
        alert("Please enter a valid label (A-Z)");
        return;
    }

    // Send data to the Flask server
    fetch("/save", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ label, matrix })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        resetGrid();
    })
    .catch(error => console.error("Error:", error));
}

function resetGrid() {
    // Clear the grid and reset the matrix
    matrix = Array(5).fill().map(() => Array(5).fill(0));
    Array.from(grid.children).forEach(cell => cell.classList.remove("filled"));
    labelInput.value = "";
}
