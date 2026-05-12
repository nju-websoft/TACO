import json
import uuid
import sys
import os

def add_ids_to_json(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from '{file_path}'.\n{e}")
        return
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading '{file_path}'.\n{e}")
        return

    if not isinstance(data, list):
        print(f"Error: The JSON content in '{file_path}' is not a list. Expected a list of objects.")
        return

    modified_count = 0
    for item in data:
        if isinstance(item, dict):
            if 'id' not in item:
                item['id'] = str(uuid.uuid4())[:8]
                modified_count += 1
        else:
            print("Warning: Found a non-dictionary item in the list. Skipping.")

    if modified_count > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Successfully added 'id' to {modified_count} items in '{file_path}'.")
        except Exception as e:
            print(f"Error: Failed to write changes to '{file_path}'.\n{e}")
    else:
        print(f"No changes made to '{file_path}'. All items already have 'id' fields.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_ids.py <json_file_path>")
    else:
        file_path = sys.argv[1]
        add_ids_to_json(file_path)
