from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT='-e .'

def get_requirements(filepath: str) -> List[str]:
    '''
    this function will return the list of requirement packages
    '''

    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
        
    return requirements

setup(
    name="job_fraud_prediction",
    version='1.0.0',
    author='Amandeep Yadav',
    author_email='ay2728850@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)