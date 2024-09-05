import re
import os
import ast
import json
import subprocess as sub

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, '_downloads')
_FILES_DIR = os.path.join(SCRIPT_DIR, '_files')

REPO = {
    "scikit-learn": "https://github.com/scikit-learn/scikit-learn",
    "numpy": "https://github.com/numpy/numpy",
    "scipy": "https://github.com/scipy/scipy",
    "pandas": "https://github.com/pandas-dev/pandas"
}

# ========== Download ==========
def download_library(lib):
    root = os.getcwd()
    os.chdir(_DOWNLOAD_DIR)
    command = f"git clone {REPO[lib]}"
    print(command)
    sub.run(command, shell=True)
    os.chdir(root)

# ========== Extract ==========
def extract(cons):
    lib = cons["lib"]
    sha = cons["sha"]
    filepath = cons["filepath"]
    modifiedfilepath = cons["filepath"].replace("/", "--")
    cls = cons["class"]
    func = cons["func"]
    docfrom = cons["docfrom"]

    root = os.getcwd()
    os.chdir(f"{_DOWNLOAD_DIR}/{lib}")
    sub.run(f"git checkout -f main", shell=True, stdout=sub.DEVNULL, stderr=sub.DEVNULL)
    sub.run(f"git checkout {sha}", shell=True, stdout=sub.DEVNULL, stderr=sub.DEVNULL)


    extract_content(lib, sha, filepath, modifiedfilepath, cls, func, docfrom)
    os.chdir(root)


def extract_content(lib, sha, filepath, modifiedfilepath, cls, func, docfrom):    
    with open(f"{_DOWNLOAD_DIR}/{lib}/{filepath}", 'r') as f:
        content = f.read()
    f.close()
    if filepath == "scikits/learn/bayes/bayes.py":
        return

    flag = "Parameters\n-----"
    
    class ClassAndFunctionVisitor(ast.NodeVisitor):
        def __init__(self, cls, func, docfrom):
            self.cls = cls
            self.func = func
            self.docfrom = docfrom

        def visit_ClassDef(self, node):
            docstring = ast.get_docstring(node)
            if docstring and flag in docstring:
                node.decorator_list = []
                if not os.path.exists(f'{_FILES_DIR}/{lib}/{sha}'):
                    os.makedirs(f'{_FILES_DIR}/{lib}/{sha}')
                if node.name == self.cls:
                    with open(f'{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{node.name}.py', 'w') as fc:
                        fc.write(ast.unparse(node))
                if node.name == self.docfrom:
                    with open(f'{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{self.cls}_docstring.txt', 'w') as fd:
                        fd.write(docstring)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if not isinstance(node.parent, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if docstring and flag in docstring:
                    node.decorator_list = []
                    if not os.path.exists(f'{_FILES_DIR}/{lib}/{sha}'):
                        os.makedirs(f'{_FILES_DIR}/{lib}/{sha}')
                    if node.name == func:
                        with open(f'{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{node.name}.py', 'w') as fc:
                            fc.write(ast.unparse(node))
                    if node.name == self.docfrom:
                        with open(f'{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{self.func}_docstring.txt', 'w') as fd:
                            fd.write(docstring)
            self.generic_visit(node) # continue to traverse nodes

    def add_parent_info(node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
            add_parent_info(child)

    tree = ast.parse(content)
    add_parent_info(tree)
    visitor = ClassAndFunctionVisitor(cls, func, docfrom)
    visitor.visit(tree)
    
# ========== Find ==========
def findpa(cons):
    lib = cons["lib"]
    sha = cons["sha"]
    modifiedfilepath = cons["filepath"].replace("/", "--")
    cls = cons["class"]
    func = cons["func"]

    if cls == "NA":
        doc_path = f"{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{func}_docstring.txt"
        output_path = f"{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{func}_pa.json"
    else:
        doc_path = f"{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{cls}_docstring.txt"
        output_path = f"{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{cls}_pa.json"
    parse_and_save(lib, doc_path, output_path)

def parse_and_save(project, doc_path, output_path):
    try:
        args, attributes = parse_numpy_style_docstring(doc_path)
    except:
        print("Format Wrong")
        return
    try:
        if not args:
            return 
    except:
        return 
    save_to_json(project, args, attributes, output_path)

def save_to_json(project, args, attributes, output_path):
    def remove_keys_with_stars(input_dict):
        # filter out * . and ' '
        keys_to_remove = [key for key in input_dict.keys() if '*' in key or (' ' in key and ',' not in key) or '.' in key or '[' in key or ']' in key]
        for key in keys_to_remove:
            del input_dict[key]
        return input_dict
    
    def clean_dict_keys(input_dict):
        cleaned_dict = {}
        for key, value in input_dict.items():
            # 查找括号的位置
            paren_index = key.find('(')
            if paren_index > 0:
                # 只保留括号前的部分
                new_key = key[:paren_index].strip()
            else:
                new_key = key
            cleaned_dict[new_key] = value
        return cleaned_dict

    # args = clean_dict_keys(remove_keys_with_stars(args))
    # attributes = clean_dict_keys(remove_keys_with_stars(attributes))
    args = remove_keys_with_stars(clean_dict_keys(args))
    attributes = remove_keys_with_stars(clean_dict_keys(attributes))
    
    data = {
        "param": args,
        "attr": attributes,
        "pa": {**args, **attributes}
    }

    def split_keys_with_commas(data):
        pattern1 = r'\{([^\{\}]+)\}_([a-zA-Z0-9_]+)' # {xxx, yyy}_zzz
        pattern2 = r'\(([^()]+)\)' # (xxx, yyy)
        if isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                if ',' in key:
                    print(f"key: {key}")
                    matches1 = re.findall(pattern1, key)
                    matches2 = re.findall(pattern2, key)
                    print(f"matches2: {matches2}")
                    if matches1:
                        items, suffix = matches1[0]
                        items = items.split(',')
                        items = [item.strip() for item in items]
                        expanded_items = [f"{item}_{suffix}" for item in items]
                        for ei in expanded_items:
                            new_data[ei.strip()] = value
                    elif matches2:
                        if project == "scipy" or project == "example":
                            items = matches2[0].split(',')
                            items = [item.strip() for item in items]
                            combine_key = '_and_'.join(items)
                            new_data[combine_key.strip()] = value
                        else:
                            items = matches2[0].split(',')
                            items = [item.strip() for item in items]
                            for i in items:
                                new_data[i.strip()] = value
                    else:
                        keys = key.split(',')
                        for k in keys:
                            new_data[k.strip()] = value
                else:
                    new_data[key] = split_keys_with_commas(value)
            return new_data
        elif isinstance(data, list):
            return [split_keys_with_commas(item) for item in data]
        else:
            return data
    
    new_data = split_keys_with_commas(data)
        

    with open(output_path, 'w') as json_file:
        json.dump(new_data, json_file, indent=4)


def parse_numpy_style_docstring(file_path):
    args = {}
    attributes = {}
    section = None
    current_key = None
    current_value = []

    first_arg = False
    leading_space = 0

    def save_current_param():
        nonlocal current_key, current_value
        if current_key is not None:
            combined_value = " ".join(current_value).strip()
            combined_value = re.sub(r'\s*,\s*', ', ', combined_value)  # Ensure spaces after commas
            if section == "args":
                args[current_key] = combined_value
            elif section == "attributes":
                attributes[current_key] = combined_value
            current_key = None
            current_value = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip()
            if line.startswith("Parameters"):
                next(file)
                save_current_param()
                section = "args"
            elif line.startswith("Attributes"):
                next(file)
                save_current_param()
                section = "attributes"
            elif line.startswith("Returns"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("return"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Examples"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Example"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Yields"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Notes"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("See Also"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Raises"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("References"):
                next(file)
                save_current_param()
                section = None
            elif section and ":" in line:
                leading_space = len(line) - len(line.lstrip(" "))
                if not first_arg:
                    first_arg_leading_space = leading_space
                    first_arg = True
                
                if leading_space == first_arg_leading_space: 
                    save_current_param()
                    current_key, value = line.split(":", 1)
                    current_key = current_key.strip()
                    current_value.append(value.strip())
                else:
                    current_value.append(line.strip())
            elif section and current_key is not None:
                current_value.append(line.strip())

    save_current_param()  # Save the last parameter
    return args, attributes





if __name__ in "__main__":
    with open('bench.json', 'r', encoding='utf-8') as file:
        constraints = json.load(file)

    if not os.path.exists(_DOWNLOAD_DIR):
        os.mkdir(_DOWNLOAD_DIR)
    if not os.path.exists(f"{_FILES_DIR}"):
        os.mkdir(_FILES_DIR)

    for constraint in constraints:
        lib = constraint["lib"]
        
        if not os.path.exists(f"{_DOWNLOAD_DIR}/{lib}"):
            download_library(lib)   
        
        extract(constraint)
    
        findpa(constraint)
        