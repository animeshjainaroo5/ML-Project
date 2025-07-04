from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = '-e .'

def get_req(file_path:str)->List[str]:
    #returns the list of requirements 
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements


setup(
    name="ML Project",
    version="0.0.1",
    author="Animesh Jain",
    author_email="animeshjain.chd@gmail.com",
    packages = find_packages(),
    install_requires = get_req('requirements.txt')
)
