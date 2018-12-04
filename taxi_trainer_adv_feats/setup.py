# minimal setup based on
# https://python-packaging.readthedocs.io/en/latest/minimal.html

from setuptools import setup

setup(name='taxi_trainer_adv_feats',
      version='0.1',
      description='test of tensorflow distributed training, taxifare dataset, advanced features',
      url='http://tensorflow.org',
      author='Michal Kosinski',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['trainer'],
      zip_safe=False)
