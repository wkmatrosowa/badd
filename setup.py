from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    readme = f.read()

setup(name='badd',
      description='In English',
      package_dir={"": "badd"},
      packages=find_packages(where="badd"),
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/wksmirnowa/badd',
      author='Vladislava Smirnova',
      author_email='wksmirnowa@gmail.com',
      license='MIT',
      version='1.0.24',
      python_requires=">=3.7.*",
      install_requires=['gensim==3.8.1',
                        'torch==1.8.1',
                        'nltk==3.2.5',
                        'emoji==1.2.0',
                        'pymorphy2==0.9.1'],
      keywords='toxic ad advertising obscene classification texts words nlp nn'
      )
