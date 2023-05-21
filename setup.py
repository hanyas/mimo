from setuptools import setup

setup(name='mimo',
      version='0.1.0',
      description='Mixture models inference',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 
                        'scikit-learn', 'pandas',
                        'matplotlib', 'tikzplotlib',
                        'pathos', 'tqdm'],
      packages=['mimo'],
      zip_safe=False,
      )
