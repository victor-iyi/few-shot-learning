"""Helper class for filtering TensorFlow's version imoprt warnings.

   @description
     For supressing TensorFlow's import warnings.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     Package: omniglot
     File: supress.py
     Created on 1 August, 2018 @ 10:47 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

from warnings import filterwarnings

filterwarnings('ignore')
