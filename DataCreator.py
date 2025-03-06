from PIL import Image
import os
import random

seed_dir = "data/seedOnly"        # Directory containing seed images 
background_dir = "data/backgrounds"    # Directory containing background images

# Define the output directory where the composite image will be saved.
output_dir = 'data/syntheticData'
os.makedirs(output_dir, exist_ok=True)   # Create the output directory if it doesn't exist

num_seeds = int(input("Enter the number of seeds to place in the composite image: "))

seed_files = [os.path.join(seed_dir, f) for f in os.listdir(seed_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
background_files = [os.path.join(background_dir, f) for f in os.listdir(background_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not seed_files:
    print("No seed images found in the seed directory.")
    exit(1)
if not background_files:
    print("No background images found in the background directory.")
    exit(1)
  
bg_file = random.choice(background_files)
background = Image.open(bg_file).convert('RGBA')
bg_width, bg_height = background.size

# Overlay the specified number of seeds on the background.
for i in range(num_seeds):
    # Choose a random seed image.
    seed_file = random.choice(seed_files)
    seed_img = Image.open(seed_file).convert('RGBA')
    
    # Optional: Randomly scale the seed image for variation.
    scale_factor = random.uniform(0.5, 1.5)  # Scale between 50% and 150%
    seed_width, seed_height = seed_img.size
    new_seed_size = (int(seed_width * scale_factor / 5), int(seed_height * scale_factor / 5))
    seed_img = seed_img.resize(new_seed_size)
    
    # Ensure the seed image fits on the background.
    if seed_img.width > bg_width or seed_img.height > bg_height:
        # Resize seed to fit if it is too large.
        scale_factor = min(bg_width / seed_img.width, bg_height / seed_img.height) * 0.9
        new_seed_size = (int(seed_img.width * scale_factor), int(seed_img.height * scale_factor))
        seed_img = seed_img.resize(new_seed_size)
    
    # Calculate random position ensuring the seed fits entirely on the background.
    max_x = bg_width - seed_img.width
    max_y = bg_height - seed_img.height
    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)
    
    # Paste the seed image onto the background using its alpha channel as mask.
    background.paste(seed_img, (pos_x, pos_y), seed_img)

fileName = "image" + str(random.randint(0,9999999)) + ".png"

# Save the composite image as a PNG file.
output_filename = os.path.join(output_dir, fileName)
background.save(output_filename)
print(f"Image saved as {output_filename}")