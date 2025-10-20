from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import random
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from torch.utils.data import Subset
from rdkit.Chem import BRICS
from torch_geometric.data import Batch, Data
import torch
from rdkit.Chem import Lipinski
from pubchemfp import GetPubChemFPs

def get_mol_property(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        logp = Descriptors.MolLogP(mol)
    except:
        logp = 0

    try:
        tpsa = Descriptors.TPSA(mol)
    except:
        tpsa = 0

    try:
        mw = Descriptors.MolWt(mol)
    except:
        mw = 0

    return [logp, tpsa, mw]

def get_maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_fp = np.array([int(bit) for bit in maccs_fp.ToBitString()])
    return maccs_fp

def get_pubchem_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pubchem_fp = GetPubChemFPs(mol)
    return pubchem_fp

def get_erg_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    erg_fp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    return erg_fp

x_map = {
    'atomic_num':list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':list(range(0, 11)),
    'formal_charge':list(range(-5, 7)),
    'num_hs':list(range(0, 9)),
    'num_radical_electrons':list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

def get_atom_feature_dims():
    return list(map(len, [
        x_map['atomic_num'],
        x_map['chirality'],
        x_map['degree'],
        x_map['formal_charge'],
        x_map['num_hs'],
        x_map['num_radical_electrons'],
        x_map['hybridization'],
        x_map['is_aromatic'],
        x_map['is_in_ring']
        ]))

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

def get_bond_feature_dims():
    return list(map(len, [
        e_map['bond_type'],
        e_map['stereo'],
        e_map['is_conjugated']
        ]))

def get_atom_and_bond_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x = []
    for atom in mol.GetAtoms():
        row = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        x.append(row)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_index += [[i, j], [j, i]]
        edge_attr += [e, e]

    return x, edge_index, edge_attr

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(smiles_list, frac_valid=0.1, frac_test=0.1, balanced=False, seed=0):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """

    frac_train = 1 - frac_valid - frac_test
    train_size, val_size, test_size = \
        frac_train * len(smiles_list), frac_valid * len(smiles_list), frac_test * len(smiles_list)

    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        try:
            scaffold = generate_scaffold(smiles, include_chirality=True)
        except:
            continue
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(all_scaffolds.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        all_scaffold_sets = big_index_sets + small_index_sets
    else:
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    return train_idx, valid_idx, test_idx

def random_split(dataset, task_idx=None, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        y_task = np.array([data.label[task_idx] for data in dataset])
        non_null = ~np.isnan(y_task)
        idx_array = np.where(non_null)[0]
        dataset = Subset(dataset, idx_array)
    else:
        pass

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    return train_idx, valid_idx, test_idx

smiles_vocab = {'[PAD]': 0, 'c': 1, 'C': 2, '(': 3, ')': 4, 'O': 5, '1': 6, '2': 7, '=': 8, 'N': 9, '.': 10, 'n': 11, '3': 12, 'F': 13, 'Cl': 14, '>>': 15, '~': 16, '-': 17, '4': 18, '[C@H]': 19, 'S': 20, '[C@@H]': 21, '[O-]': 22, 'Br': 23, '#': 24, '/': 25, '[nH]': 26, '[N+]': 27, 's': 28, '5': 29, 'o': 30, 'P': 31, '[Na+]': 32, '[Si]': 33, 'I': 34, '[Na]': 35, '[Pd]': 36, '[K+]': 37, '[K]': 38, '[P]': 39, 'B': 40, '[C@]': 41, '[C@@]': 42, '[Cl-]': 43, '6': 44, '[OH-]': 45, '\\': 46, '[N-]': 47, '[Li]': 48, '[H]': 49, '[2H]': 50, '[NH4+]': 51, '[c-]': 52, '[P-]': 53, '[Cs+]': 54, '[Li+]': 55, '[Cs]': 56, '[NaH]': 57, '[H-]': 58, '[O+]': 59, '[BH4-]': 60, '[Cu]': 61, '7': 62, '[Mg]': 63, '[Fe+2]': 64, '[n+]': 65, '[Sn]': 66, '[BH-]': 67, '[Pd+2]': 68, '[CH]': 69, '[I-]': 70, '[Br-]': 71, '[C-]': 72, '[Zn]': 73, '[B-]': 74, '[F-]': 75, '[Al]': 76, '[P+]': 77, '[BH3-]': 78, '[Fe]': 79, '[C]': 80, '[AlH4]': 81, '[Ni]': 82, '[SiH]': 83, '8': 84, '[Cu+2]': 85, '[Mn]': 86, '[AlH]': 87, '[nH+]': 88, '[AlH4-]': 89, '[O-2]': 90, '[Cr]': 91, '[Mg+2]': 92, '[NH3+]': 93, '[S@]': 94, '[Pt]': 95, '[Al+3]': 96, '[S@@]': 97, '[S-]': 98, '[Ti]': 99, '[Zn+2]': 100, '[PH]': 101, '[NH2+]': 102, '[Ru]': 103, '[Ag+]': 104, '[S+]': 105, '[I+3]': 106, '[NH+]': 107, '[Ca+2]': 108, '[Ag]': 109, '9': 110, '[Os]': 111, '[Se]': 112, '[SiH2]': 113, '[Ca]': 114, '[Ti+4]': 115, '[Ac]': 116, '[Cu+]': 117, '[S]': 118, '[Rh]': 119, '[Cl+3]': 120, '[cH-]': 121, '[Zn+]': 122, '[O]': 123, '[Cl+]': 124, '[SH]': 125, '[H+]': 126, '[Pd+]': 127, '[se]': 128, '[PH+]': 129, '[I]': 130, '[Pt+2]': 131, '[C+]': 132, '[Mg+]': 133, '[Hg]': 134, '[W]': 135, '[SnH]': 136, '[SiH3]': 137, '[Fe+3]': 138, '[NH]': 139, '[Mo]': 140, '[CH2+]': 141, '%10': 142, '[CH2-]': 143, '[CH2]': 144, '[n-]': 145, '[Ce+4]': 146, '[NH-]': 147, '[Co]': 148, '[I+]': 149, '[PH2]': 150, '[Pt+4]': 151, '[Ce]': 152, '[B]': 153, '[Sn+2]': 154, '[Ba+2]': 155, '%11': 156, '[Fe-3]': 157, '[18F]': 158, '[SH-]': 159, '[Pb+2]': 160, '[Os-2]': 161, '[Zr+4]': 162, '[N]': 163, '[Ir]': 164, '[Bi]': 165, '[Ni+2]': 166, '[P@]': 167, '[Co+2]': 168, '[s+]': 169, '[As]': 170, '[P+3]': 171, '[Hg+2]': 172, '[Yb+3]': 173, '[CH-]': 174, '[Zr+2]': 175, '[Mn+2]': 176, '[CH+]': 177, '[In]': 178, '[KH]': 179, '[Ce+3]': 180, '[Zr]': 181, '[AlH2-]': 182, '[OH2+]': 183, '[Ti+3]': 184, '[Rh+2]': 185, '[Sb]': 186, '[S-2]': 187, '%12': 188, '[P@@]': 189, '[Si@H]': 190, '[Mn+4]': 191, 'p': 192, '[Ba]': 193, '[NH2-]': 194, '[Ge]': 195, '[Pb+4]': 196, '[Cr+3]': 197, '[Au]': 198, '[LiH]': 199, '[Sc+3]': 200, '[o+]': 201, '[Rh-3]': 202, '%13': 203, '[Br]': 204, '[Sb-]': 205, '[S@+]': 206, '[I+2]': 207, '[Ar]': 208, '[V]': 209, '[Cu-]': 210, '[Al-]': 211, '[Te]': 212, '[13c]': 213, '[13C]': 214, '[Cl]': 215, '[PH4+]': 216, '[SiH4]': 217, '[te]': 218, '[CH3-]': 219, '[S@@+]': 220, '[Rh+3]': 221, '[SH+]': 222, '[Bi+3]': 223, '[Br+2]': 224, '[La]': 225, '[La+3]': 226, '[Pt-2]': 227, '[N@@]': 228, '[PH3+]': 229, '[N@]': 230, '[Si+4]': 231, '[Sr+2]': 232, '[Al+]': 233, '[Pb]': 234, '[SeH]': 235, '[Si-]': 236, '[V+5]': 237, '[Y+3]': 238, '[Re]': 239, '[Ru+]': 240, '[Sm]': 241, '*': 242, '[3H]': 243, '[NH2]': 244, '[Ag-]': 245, '[13CH3]': 246, '[OH+]': 247, '[Ru+3]': 248, '[OH]': 249, '[Gd+3]': 250, '[13CH2]': 251, '[In+3]': 252, '[Si@@]': 253, '[Si@]': 254, '[Ti+2]': 255, '[Sn+]': 256, '[Cl+2]': 257, '[AlH-]': 258, '[Pd-2]': 259, '[SnH3]': 260, '[B+3]': 261, '[Cu-2]': 262, '[Nd+3]': 263, '[Pb+3]': 264, '[13cH]': 265, '[Fe-4]': 266, '[Ga]': 267, '[Sn+4]': 268, '[Hg+]': 269, '[11CH3]': 270, '[Hf]': 271, '[Pr]': 272, '[Y]': 273, '[S+2]': 274, '[Cd]': 275, '[Cr+6]': 276, '[Zr+3]': 277, '[Rh+]': 278, '[CH3]': 279, '[N-3]': 280, '[Hf+2]': 281, '[Th]': 282, '[Sb+3]': 283, '%14': 284, '[Cr+2]': 285, '[Ru+2]': 286, '[Hf+4]': 287, '[14C]': 288, '[Ta]': 289, '[Tl+]': 290, '[B+]': 291, '[Os+4]': 292, '[PdH2]': 293, '[Pd-]': 294, '[Cd+2]': 295, '[Co+3]': 296, '[S+4]': 297, '[Nb+5]': 298, '[123I]': 299, '[c+]': 300, '[Rb+]': 301, '[V+2]': 302, '[CH3+]': 303, '[Ag+2]': 304, '[cH+]': 305, '[Mn+3]': 306, '[Se-]': 307, '[As-]': 308, '[Eu+3]': 309, '[SH2]': 310, '[Sm+3]': 311, '[IH+]': 312, '%15': 313, '[OH3+]': 314, '[PH3]': 315, '[IH2+]': 316, '[SH2+]': 317, '[Ir+3]': 318, '[AlH3]': 319, '[Sc]': 320, '[Yb]': 321, '[15NH2]': 322, '[Lu]': 323, '[sH+]': 324, '[Gd]': 325, '[18F-]': 326, '[SH3+]': 327, '[SnH4]': 328, '[TeH]': 329, '[Si@@H]': 330, '[Ga+3]': 331, '[CaH2]': 332, '[Tl]': 333, '[Ta+5]': 334, '[GeH]': 335, '[Br+]': 336, '[Sr]': 337, '[Tl+3]': 338, '[Sm+2]': 339, '[PH5]': 340, '%16': 341, '[N@@+]': 342, '[Au+3]': 343, '[C-4]': 344, '[Nd]': 345, '[Ti+]': 346, '[IH]': 347, '[N@+]': 348, '[125I]': 349, '[Eu]': 350, '[Sn+3]': 351, '[Nb]': 352, '[Er+3]': 353, '[123I-]': 354, '[14c]': 355, '%17': 356, '[SnH2]': 357, '[YH]': 358, '[Sb+5]': 359, '[Pr+3]': 360, '[Ir+]': 361, '[N+3]': 362, '[AlH2]': 363, '[19F]': 364, '%18': 365, '[Tb]': 366, '[14CH]': 367, '[Mo+4]': 368, '[Si+]': 369, '[BH]': 370, '[Be]': 371, '[Rb]': 372, '[pH]': 373, '%19': 374, '%20': 375, '[Xe]': 376, '[Ir-]': 377, '[Be+2]': 378, '[C+4]': 379, '[RuH2]': 380, '[15NH]': 381, '[U+2]': 382, '[Au-]': 383, '%21': 384, '%22': 385, '[Au+]': 386, '[15n]': 387, '[Al+2]': 388, '[Tb+3]': 389, '[15N]': 390, '[V+3]': 391, '[W+6]': 392, '[14CH3]': 393, '[Cr+4]': 394, '[ClH+]': 395, 'b': 396, '[Ti+6]': 397, '[Nd+]': 398, '[Zr+]': 399, '[PH2+]': 400, '[Fm]': 401, '[N@H+]': 402, '[RuH]': 403, '[Dy+3]': 404, '%23': 405, '[Hf+3]': 406, '[W+4]': 407, '[11C]': 408, '[13CH]': 409, '[Er]': 410, '[124I]': 411, '[LaH]': 412, '[F]': 413, '[siH]': 414, '[Ga+]': 415, '[Cm]': 416, '[GeH3]': 417, '[IH-]': 418, '[U+6]': 419, '[SeH+]': 420, '[32P]': 421, '[SeH-]': 422, '[Pt-]': 423, '[Ir+2]': 424, '[se+]': 425, '[U]': 426, '[F+]': 427, '[BH2]': 428, '[As+]': 429, '[Cf]': 430, '[ClH2+]': 431, '[Ni+]': 432, '[TeH3]': 433, '[SbH2]': 434, '[Ag+3]': 435, '%24': 436, '[18O]': 437, '[PH4]': 438, '[Os+2]': 439, '[Na-]': 440, '[Sb+2]': 441, '[V+4]': 442, '[Ho+3]': 443, '[68Ga]': 444, '[PH-]': 445, '[Bi+2]': 446, '[Ce+2]': 447, '[Pd+3]': 448, '[99Tc]': 449, '[13C@@H]': 450, '[Fe+6]': 451, '[c]': 452, '[GeH2]': 453, '[10B]': 454, '[Cu+3]': 455, '[Mo+2]': 456, '[Cr+]': 457, '[Pd+4]': 458, '[Dy]': 459, '[AsH]': 460, '[Ba+]': 461, '[SeH2]': 462, '[In+]': 463, '[TeH2]': 464, '[BrH+]': 465, '[14cH]': 466, '[W+]': 467, '[13C@H]': 468, '[AsH2]': 469, '[In+2]': 470, '[N+2]': 471, '[N@@H+]': 472, '[SbH]': 473, '[60Co]': 474, '[AsH4+]': 475, '[AsH3]': 476, '[18OH]': 477, '[Ru-2]': 478, '[Na-2]': 479, '[CuH2]': 480, '[31P]': 481, '[Ti+5]': 482, '[35S]': 483, '[P@@H]': 484, '[ArH]': 485, '[Co+]': 486, '[Zr-2]': 487, '[BH2-]': 488, '[131I]': 489, '[SH5]': 490, '[VH]': 491, '[B+2]': 492, '[Yb+2]': 493, '[14C@H]': 494, '[211At]': 495, '[NH3+2]': 496, '[IrH]': 497, '[IrH2]': 498, '[Rh-]': 499, '[Cr-]': 500, '[Sb+]': 501, '[Ni+3]': 502, '[TaH3]': 503, '[Tl+2]': 504, '[64Cu]': 505, '[Tc]': 506, '[Cd+]': 507, '[1H]': 508, '[15nH]': 509, '[AlH2+]': 510, '[FH+2]': 511, '[BiH3]': 512, '[Ru-]': 513, '[Mo+6]': 514, '[AsH+]': 515, '[BaH2]': 516, '[BaH]': 517, '[Fe+4]': 518, '[229Th]': 519, '[Th+4]': 520, '[As+3]': 521, '[NH+3]': 522, '[P@H]': 523, '[Li-]': 524, '[7NaH]': 525, '[Bi+]': 526, '[PtH+2]': 527, '[p-]': 528, '[Re+5]': 529, '[NiH]': 530, '[Ni-]': 531, '[Xe+]': 532, '[Ca+]': 533, '[11c]': 534, '[Rh+4]': 535, '[AcH]': 536, '[HeH]': 537, '[Sc+2]': 538, '[Mn+]': 539, '[UH]': 540, '[14CH2]': 541, '[SiH4+]': 542, '[18OH2]': 543, '[Ac-]': 544, '[Re+4]': 545, '[118Sn]': 546, '[153Sm]': 547, '[P+2]': 548, '[9CH]': 549, '[9CH3]': 550, '[Y-]': 551, '[NiH2]': 552, '[Si+2]': 553, '[Mn+6]': 554, '[ZrH2]': 555, '[C-2]': 556, '[Bi+5]': 557, '[24NaH]': 558, '[Fr]': 559, '[15CH]': 560, '[Se+]': 561, '[At]': 562, '[P-3]': 563, '[124I-]': 564, '[CuH2-]': 565, '[Nb+4]': 566, '[Nb+3]': 567, '[MgH]': 568, '[Ir+4]': 569, '[67Ga+3]': 570, '[67Ga]': 571, '[13N]': 572, '[15OH2]': 573, '[2NH]': 574, '[Ho]': 575, '[Cn]': 576,'[UNK]': 577,'[MASK]':578}

def match_smiles(smiles_str, max_length=200):
    result = []
    i = 0
    while i < len(smiles_str):
        for j in range(len(smiles_str), i, -1):
            substring = smiles_str[i:j]
            if substring in smiles_vocab:
                result.append(smiles_vocab[substring])
                i = j
                break
        else:
            result.append(smiles_vocab['[UNK]'])
            i += 1

    if len(result) > max_length:
        result = result[:max_length]
    elif len(result) < max_length:
        pad_index = smiles_vocab['[PAD]']
        result.extend([pad_index] * (max_length - len(result)))

    return result

def get_substructures(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fragments = BRICS.BRICSDecompose(mol,singlePass=True,keepNonLeafNodes=True)
    return list(fragments)

def get_substructure_tensor(smiles):
    fragments = get_substructures(smiles)

    gnn_inputs = [get_atom_and_bond_features(frag) for frag in fragments]
    sub_graph_list = []
    for data in gnn_inputs:
        temp_data = Data(x=torch.tensor(data[0], dtype=torch.float).view(-1, 9),
                         edge_index=torch.tensor(data[1], dtype=torch.long).view(2, -1),
                         edge_attr=torch.tensor(data[2], dtype=torch.float).view(-1, 5),
                         )
        sub_graph_list.append(temp_data)
    sub_graph_list = Batch().from_data_list(sub_graph_list)
    sub_seq_list = torch.stack([torch.tensor(match_smiles(x), dtype=torch.long) for x in fragments])
    return sub_graph_list,sub_seq_list

def get_substructure_graph_tensor(smiles):
    fragments = get_substructures(smiles)

    inputs = [get_atom_and_bond_features(frag) for frag in fragments]
    sub_graph_list = []
    for data in inputs:
        temp_data = Data(x=torch.tensor(data[0], dtype=torch.long).view(-1, 9),
                         edge_index=torch.tensor(data[1], dtype=torch.long).view(2, -1),
                         edge_attr=torch.tensor(data[2], dtype=torch.long).view(-1, 3),
                         )
        sub_graph_list.append(temp_data)
    sub_graph_list = Batch().from_data_list(sub_graph_list)
    return sub_graph_list

def get_substructure_seq_tensor(smiles):
    fragments = get_substructures(smiles)
    sub_seq_list = torch.stack([torch.tensor(match_smiles(x), dtype=torch.long) for x in fragments])
    return sub_seq_list

if __name__ == "__main__":

    pass