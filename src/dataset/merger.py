"""Utility to merge Spark output part files into a single CSV."""
import os
import glob


class CsvMerger:
    """Merges Spark's partitioned CSV output into a single file."""
    
    def __init__(self, parts_folder: str, output_file: str):
        """
        Initialize the merger.
        
        Args:
            parts_folder: Path to the folder containing Spark part files.
            output_file: Path to the merged output CSV file.
        """
        self.parts_folder = parts_folder
        self.output_file = output_file
    
    def merge(self) -> str:
        """
        Merge all part files into a single CSV file.
        
        Returns:
            Path to the merged file.
        """
        # Find all part files (they start with 'part-' and end with '.csv')
        part_files = sorted(glob.glob(os.path.join(self.parts_folder, 'part-*.csv')))
        
        if not part_files:
            raise FileNotFoundError(f"No part files found in {self.parts_folder}")
        
        print(f"Found {len(part_files)} part files to merge...")
        
        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            for i, part_file in enumerate(part_files):
                with open(part_file, 'r', encoding='utf-8') as infile:
                    if i == 0:
                        # First file: include header
                        outfile.write(infile.read())
                    else:
                        # Subsequent files: skip header line
                        lines = infile.readlines()
                        if lines:
                            outfile.writelines(lines[1:])
        
        print(f"Merged CSV saved to: {self.output_file}")
        return self.output_file
