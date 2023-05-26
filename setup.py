from setuptools import setup, find_packages

setup(
    name='motlee',
    version='0.1.0',    
    description='Multiple Object Tracking with Localization Error Elimination',
    url='https://gitlab.com/mit-acl/dmot/motlee',
    author='Mason Peterson, Parker Lusk',
    author_email='masonbp@mit.edu, parkerclusk@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy',
                        'matplotlib',
                        'scipy',
                        'pandas',
                        'gtsam',
                        'pyyaml',
                        'opencv-python',
                        'tqdm',
                        'motmetrics',
                        'bagpy',
                        'open3d==0.17.0'
                      ],

    # classifiers=[
    #     'Development Status :: 1 - Planning',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: BSD License',  
    #     'Operating System :: POSIX :: Linux',        
    #     'Programming Language :: Python :: 2',
    #     'Programming Language :: Python :: 2.7',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.4',
    #     'Programming Language :: Python :: 3.5',
    # ],
)