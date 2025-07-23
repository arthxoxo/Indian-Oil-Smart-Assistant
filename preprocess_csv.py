import re

input_file = "retail_outlets.csv"
output_file = "retail_outlets_cleaned.csv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    buffer = ""
    for line in infile:
        # If the line starts with a quote and a number, it's a new record
        if re.match(r'^"\\d+', line.strip()):
            if buffer:
                outfile.write(buffer)
            buffer = line
        else:
            # Join multi-line records
            buffer = buffer.rstrip('\n') + " " + line.lstrip()
    # Write the last buffer
    if buffer:
        outfile.write(buffer)

print(f"Preprocessing complete! Cleaned file saved as {output_file}")
