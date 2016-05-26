import os
import inspect

module_names = []
for subdir, dirs, files in os.walk('./'):
    for file in files:
        fileName, fileExtension = os.path.splitext(file)
        if fileExtension == '.py' and fileName != '__init__' and fileName != 'launch_test' and fileName != 'mocks':
            module_names.append(fileName)
    
modules = map(__import__, module_names)
for module in modules:
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            d = dir(obj)
            for d1 in d:
                if not d1.find("test"):
                    function_name = module.__name__ + '.' + obj.__name__ + '.' + d1
                    bash_command = "python -m unittest " + function_name
                    print(bash_command)
                    os.system(bash_command)
