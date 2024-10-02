from setuptools import setup, find_packages

setup(
    name='tcaff',
    version='0.1.0',    
    description='Multiple Object Tracking with Localization Error Elimination',
    url='https://github.com/mit-acl/tcaff',
    author='Mason Peterson, Parker Lusk',
    author_email='masonbp@mit.edu, parkerclusk@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy',
                        'matplotlib',
                        'scipy',
                        'pyyaml',
                        'opencv-python',
                        'tqdm',
                        'motmetrics',
                        'yolov7-package',
                        'scikit-image',
                        'fastsam @ git+ssh://git@github.com/CASIA-IVA-Lab/FastSAM@4d153e9',
                        'robotdatapy @ git+ssh://git@github.com/mbpeterson70/robotdatapy@0e7853d',
                        'plot_utils @ git+ssh://git@github.com/mbpeterson70/plot_utils@fab133e',
                      ],
    extras_require={
        'align': ['open3d==0.17.0',]
    },

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
