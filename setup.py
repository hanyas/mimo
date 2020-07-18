from setuptools import setup

setup(name='mimo',
      version='0.0.1',
      description='Mixture models inference',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib', 'pandas',
                        'tikzplotlib', 'pathos', 'future', 'sklearn'],
      packages=['mimo'],
      zip_safe=False,
      )
