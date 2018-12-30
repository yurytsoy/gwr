from setuptools import setup

setup(name='gwr',
      version='0.1',
      description='Implementation of the Grow When Required network',
      url='http://github.com/yurytsoy/gwr',
      author='Yury Tsoy',
      author_email='yurytsoy@gmail.com',
      license='MIT',
      packages=['gwr'],
      install_requires=[
          'numpy',
          'jsonpickle'
      ],
      zip_safe=False)
