import zipfile
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def extract(source_dir='raw', target_dir='extracted'):
    # Iterate through all zip files in the specified directory
    zip_files = list(Path(source_dir).glob('*.zip'))
    for zip_path in tqdm(zip_files, desc="Extracting"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find the file named 'Measurement' regardless of which subfolder it is in
                target_entry = next((f for f in zf.namelist() if f.split('/')[-1] == 'Measurement'), None)
                
                if target_entry:
                    # Format: {original_zip_name}.json
                    output_name = f"{zip_path.stem}.json"
                    output_path = Path(target_dir) / output_name
                    
                    # Extract the content and write it directly to the new filename
                    with open(output_path, 'wb') as f:
                        f.write(zf.read(target_entry))
                else:
                    tqdm.write(f"Warning: No 'Measurement' file found in {zip_path.name}")
        except Exception as e:
            tqdm.write(f"Error extracting {zip_path.name}: {e}")

def combine(source_dir='extracted', output_file='combined/combined.json'):
    source_path = Path(source_dir)
    out_file = Path(output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    json_files = list(source_path.glob('*.json'))
    with open(out_file, 'w', encoding='utf-8') as outf:
        outf.write('[\n')
        first_entry = True
        
        for json_file in tqdm(json_files, desc="Combining"):
            try:
                with open(json_file, 'r', encoding='utf-8') as inf:
                    data = json.load(inf)
                    if isinstance(data, list):
                        for item in data:
                            if not first_entry:
                                outf.write(',\n')
                            json.dump(item, outf)
                            first_entry = False
                    else:
                        tqdm.write(f"Skipping {json_file.name}: root element is not a list.")
            except json.JSONDecodeError:
                tqdm.write(f"Skipping {json_file.name}: invalid JSON.")
            except Exception as e:
                tqdm.write(f"Error processing {json_file.name}: {e}")
        
        outf.write('\n]')

def main():
    parser = argparse.ArgumentParser(description="MobiPerf data processor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--source-dir", default="raw", help="Directory containing zip files")
    extract_parser.add_argument("--target-dir", default="extracted", help="Directory to save extracted measurements")

    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--source-dir", default="extracted", help="Directory containing extracted JSON files")
    combine_parser.add_argument("--output-file", default="combined/combined.json", help="Path to the output JSON file")

    args = parser.parse_args()

    if args.command == "extract":
        extract(source_dir=args.source_dir, target_dir=args.target_dir)
    elif args.command == "combine":
        combine(source_dir=args.source_dir, output_file=args.output_file)

if __name__ == "__main__":
    main()