import csv
import random

def load_base_patterns(filename="alphabet_data.csv"):
    """Load base patterns for each letter from a CSV file."""
    base_patterns = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            letter = row[0]
            pattern = list(map(int, row[1:]))  # Convert pattern values to integers
            base_patterns[letter] = pattern
    return base_patterns

def add_noise(pattern, noise_level=0.1):
    """Add noise to a pattern by flipping bits randomly based on noise level."""
    noisy_pattern = pattern[:]
    for i in range(len(noisy_pattern)):
        if random.random() < noise_level:
            noisy_pattern[i] = 1 - noisy_pattern[i]  # Flip 0 to 1 or 1 to 0
    return noisy_pattern

def generate_variations(base_patterns, num_variations=10, noise_level=0.1):
    """Generate multiple variations of each letter's base pattern."""
    variations = []
    for letter, pattern in base_patterns.items():
        for _ in range(num_variations):
            noisy_pattern = add_noise(pattern, noise_level)
            variations.append([letter] + noisy_pattern)
    return variations

def save_variations_to_csv(variations, filename="alphabet_data.csv"):
    """Save the generated variations to a CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        header = ["Letter"] + [f"Pixel_{i}" for i in range(1, 26)]
        writer.writerow(header)
        
        # Write variations
        writer.writerows(variations)

# Load base patterns
base_patterns = load_base_patterns("alphabet_data.csv")

# Generate variations with noise
variations = generate_variations(base_patterns, num_variations=10, noise_level=0.1)

# Save to CSV
save_variations_to_csv(variations, "alphabet_data_with_noise.csv")
