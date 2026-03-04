import os

def split_csv(input_file="train.csv", output_folder="splits", num_parts=10, part_size_mb=25):
    os.makedirs(output_folder, exist_ok=True)

    part_size_bytes = part_size_mb * 1024 * 1024
    file_count = 1
    output_file = os.path.join(output_folder, f"part_{file_count}.csv")
    out = open(output_file, "w", encoding="utf-8", newline="")
    written_size = 0

    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline()  
        out.write(header)
        written_size += len(header.encode("utf-8"))

        for line in f:
            line_size = len(line.encode("utf-8"))

      
            if written_size + line_size > part_size_bytes and file_count < num_parts:
                out.close()
                file_count += 1
                output_file = os.path.join(output_folder, f"part_{file_count}.csv")
                out = open(output_file, "w", encoding="utf-8", newline="")
                out.write(header)
                written_size = len(header.encode("utf-8"))

            out.write(line)
            written_size += line_size

    out.close()
    print(f" Split into {file_count} files in '{output_folder}'")


if __name__ == "__main__":
    split_csv()
