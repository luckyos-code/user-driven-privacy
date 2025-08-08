from .Adult import *
from .Diabetes import *

# Registry for dynamic access
SPALTEN_MODULES = {
    'adult': __import__('src.spalten.Adult', fromlist=['*']),
    'diabetes': __import__('src.spalten.Diabetes', fromlist=['*'])
} 