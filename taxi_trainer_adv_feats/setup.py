# minimal setup based on
# https://python-packaging.readthedocs.io/en/latest/minimal.html

from setuptools import setup, find_packages

setup(name='taxi_trainer_adv_feats',
      version='0.1',
      description='test of tensorflow distributed training, taxifare dataset, advanced features',
      url='http://tensorflow.org',
      author='Michal Kosinski',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'click',
      ],
      zip_safe=False)
