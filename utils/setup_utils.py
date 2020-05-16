import os


def find_requirements(setup_file):
    requirement_file = os.path.dirname(
        os.path.realpath(setup_file)) + '/requirements.txt'
    install_requires = []

    if os.path.isfile(requirement_file):
        with open(requirement_file) as f:
            install_requires = f.read().splitlines()
    return install_requires
