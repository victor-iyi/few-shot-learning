"""Helper class for filtering TensorFlow's version imoprt warnings.

   @description
     For supressing TensorFlow's import warnings.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: supress.py
     Package: omniglot
     Created on 1st August, 2018 @ 12:14 AM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

from warnings import filterwarnings

filterwarnings('ignore')
