from setuptools import setup

setup(name='mimo',
      version='0.0.1',
      description='Mixture models inference',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib', 'tikzplotlib', 'joblib', 'pathos'],
      packages=['mimo'],
      zip_safe=False,
      )