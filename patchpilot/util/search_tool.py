from fuzzysearch import find_near_matches
from patchpilot.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
)
import ast


search_string_schema = {
  "type": "function",
  "function": {
    "name": "search_string",
    "description": "Accepts a string to search for and returns the file paths where the string is found. Note: The string should be specific enough (e.g., an error message) to ensure accurate search results.",
    "parameters": {
      "type": "object",
      "properties": {
        "query_string": {
          "type": "string",
          "description": "The specific string to search for, which should be sufficiently unique."
        }
      },
      "required": ["query_string"]
    }
  }
}

search_class_def_schema = {
  "type": "function",
  "function": {
    "name": "search_class_def",
    "description": "Accepts a class name as a string and returns the file path where the class is defined.",
    "parameters": {
      "type": "object",
      "properties": {
        "class_name": {
          "type": "string",
          "description": "The name of the class to search for."
        }
      },
      "required": ["class_name"]
    }
  }
}

search_func_def_schema = {
  "type": "function",
  "function": {
    "name": "search_func_def",
    "description": "Accepts a function name as a string and returns the file path where the function is defined.",
    "parameters": {
      "type": "object",
      "properties": {
        "function_name": {
          "type": "string",
          "description": "The name of the function to search for."
        }
      },
      "required": ["function_name"]
    }
  }
}


search_func_def_with_class_and_file_schema = {
  "type": "function",
  "function": {
    "name": "search_func_def_with_class_and_file",
    "description": "Accepts a function name, file name, and class name to search and return the code of the function definition. Only function name is required, but file name and class name are preferred if available.",
    "parameters": {
      "type": "object",
      "properties": {
        "function_name": {
          "type": "string",
          "description": "The name of the function to search for.",
        },
        "class_name": {
          "type": "string",
          "description": "The name of the class containing the function, if applicable. Optional but preferred for scoped searching."
        }
      },
      "required": ["function_name"]
    }
  }
}

def search_func_def_with_class_and_file(structure, function_name: str, class_name: str = "") -> list[str]:
    """
    Accepts a function name, file name, and class name to search and return the code of the function definition, along with the file and class containing the function. Only function name is required, but file name and class name are preferred if available.
    """
    print(f"Searching for function {function_name} in class {class_name if class_name else 'any'}")
    
    found_class_name = class_name
    found_function_code = ""
    found_file_name = ""
    
    # Get the list of files, classes, and functions from the provided structure
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)

    
    for file in files:        
        file_path, file_contents = file[0], "\n".join(file[1])
        
        # Parse file contents to check for class and function definitions
        tree = ast.parse(file_contents)
        
        # If a specific class name is provided, search within the class
        if class_name:
            found_class = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    found_class = True
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == function_name:
                            found_function_code = ast.get_source_segment(file_contents, item)
                            found_file_name = file_path
                            found_class_name = class_name
                            break
                    break
            if not found_class:
                continue  # Class not found in this file, skip to next file
        else:
            # If no specific class name, search for the function definition directly
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    found_function_code = ast.get_source_segment(file_contents, node)
                    found_file_name = file_path
                    # Check if function is within a class
                    parent_class = next((cls.name for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) and node in cls.body), None)
                    found_class_name = parent_class if parent_class else ""
                    break

        if found_function_code:
          print(f"Found function {function_name} in class {found_class_name} in file {found_file_name}")
          break  # Stop search once function definition is found

    # Return the result as a list with file name, class name, and function code
    return [found_file_name, found_class_name, found_function_code]
            
      
      

def search_func_def(function_name: str, structure) -> list[str]:
    """
    Accepts a function name as a string and returns the file path where the function is defined.
    """
    search_res=[]
    print(f"searching for function {function_name}")
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    for function_struct in functions:
        if function_struct["name"] == function_name:
            search_res.append(function_struct["file"])
    return [] if not search_res else search_res


def search_class_def(class_name: str, structure) -> list[str]:
    """
    Accepts a class name as a string and returns the file path where the class is defined.
    """
    # Implementation code
    search_res=[]
    print(f"searching for class {class_name}")
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    for class_struct in classes:
        if class_struct["name"] == class_name:
            search_res.append(class_struct["file"])
    return [] if not search_res else search_res


def search_string(query_string: str, structure) -> list[str]:
    """
    Accepts a string to search for and returns the file paths where the string is found. We only return the files that contain the specific string the most number of times. 
    Note: The string should be specific enough (e.g., an error message) to ensure accurate search results.
    """
    file_to_num_occurrences = {}
    print(f"searching for string '{query_string}'")
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    for file in files:
        if query_string in "\n".join(file[1]):
            file_to_num_occurrences[file[0]] = "\n".join(file[1]).count(query_string)
    file_to_num_occurrences = dict(sorted(file_to_num_occurrences.items(), key=lambda item: item[1], reverse=True))
    if file_to_num_occurrences:
        return [file for file in file_to_num_occurrences.keys()][:20]
    #fuzzy search
     # Fuzzy search if no exact matches found
    print(f"Performing Fuzzy search for string '{query_string}'")
    fuzzy_matches = {}
    for file in files:
        file_path, file_contents = file[0], "\n".join(file[1])
        matches = find_near_matches(query_string, file_contents, max_l_dist=min(len(query_string) // 3, 5))
        if matches:
            fuzzy_matches[file_path] = len(matches)

    if fuzzy_matches:
        # Sort files by number of fuzzy matches in descending order
        sorted_fuzzy_files = sorted(fuzzy_matches.items(), key=lambda item: item[1], reverse=True)
        return [file for file, _ in sorted_fuzzy_files][:20]

    # If no matches found
    return []