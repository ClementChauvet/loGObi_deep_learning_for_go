import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='loGObi',
    version='1.0',
    author='Clément Chauvet and Télio Cropsal',
    author_email='clement.chauvet@univ-lille.fr',
    description='Implementation of loGObi a deep learning algorithm for GO with <1M parameters',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ClementChauvet/loGObi-DeepLearningForGo',
    project_urls = {
        "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    },
    license='MIT',
    packages=['loGObi', 'loGObi.src']
)