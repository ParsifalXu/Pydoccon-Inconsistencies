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

def download_library(lib):
    root = os.getcwd()
    os.chdir(_DOWNLOAD_DIR)
    command = f"git clone {REPO[lib]}"
    print(command)
    sub.run(command, shell=True)
    os.chdir(root)

def extract(cons):
    lib = cons["lib"]
    sha = cons["sha"]
    filepath = cons["filepath"]
    modifiedfilepath = cons["filepath"].replace("/", "--")
    cls = cons["class"]
    func = cons["func"]

    root = os.getcwd()
    os.chdir(f"{_DOWNLOAD_DIR}/{lib}")
    sub.run(f"git checkout -f main", shell=True, stdout=sub.DEVNULL, stderr=sub.DEVNULL)
    sub.run(f"git checkout {sha}", shell=True, stdout=sub.DEVNULL, stderr=sub.DEVNULL)


    extract_content(lib, sha, filepath, modifiedfilepath, cls, func)
    os.chdir(root)


def extract_content(lib, sha, filepath, modifiedfilepath, cls, func):    
    with open(f"{_DOWNLOAD_DIR}/{lib}/{filepath}", 'r') as f:
        content = f.read()
    f.close()
    if filepath == "scikits/learn/bayes/bayes.py":
        return

    flag = "Parameters\n-----"
    
    class ClassAndFunctionVisitor(ast.NodeVisitor):
        def __init__(self, cls, func):
            self.cls = cls
            self.func = func

        def visit_ClassDef(self, node):
            docstring = ast.get_docstring(node)
            if docstring and flag in docstring:
                node.decorator_list = []
                if not os.path.exists(f'{_FILES_DIR}/{lib}/{sha}'):
                    os.makedirs(f'{_FILES_DIR}/{lib}/{sha}')
                if node.name == cls:
                    with open(f'{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{node.name}.py', 'w') as fc:
                        fc.write(ast.unparse(node))
                    with open(f'{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{node.name}_docstring.txt', 'w') as fd:
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
                        with open(f'{_FILES_DIR}/{lib}/{sha}/{modifiedfilepath}=>{node.name}_docstring.txt', 'w') as fd:
                            fd.write(docstring)
            self.generic_visit(node) # continue to traverse nodes

    def add_parent_info(node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
            add_parent_info(child)

    tree = ast.parse(content)
    add_parent_info(tree)
    visitor = ClassAndFunctionVisitor(cls, func)
    visitor.visit(tree)
    





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
        
        