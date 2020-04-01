import os

SRC_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(SRC_DIR, '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CHILDREN_LANG_DIR = os.path.join(DATA_DIR, 'critlangacq')
DUOLINGO_LANG_DIR = os.path.join(DATA_DIR, 'duolingo')
WORDBANKR_LANG_DIR = os.path.join(DATA_DIR, 'wordbankr')
PISA2015_DIR = os.path.join(DATA_DIR, 'pisa2015_science')
GLOVE_DIR = os.path.join(DATA_DIR, 'glove')
R_DATA_DIR = os.path.join(DATA_DIR, 'R_data')
OUT_DIR = os.path.join(ROOT_DIR, 'out')

MISSING_DATA = -1  # use -1 to represent missing

IS_REAL_WORLD = {
    '1pl_simulation': False, 
    '2pl_simulation': False, 
    '1pl_nonlinear': False,
    '2pl_nonlinear': False,
    'critlangacq': True,
    'duolingo': True,
    'wordbank': True,
    'pisa2015_science': True,
}
