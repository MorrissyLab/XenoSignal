from Bio.PDB import PDBParser
import numpy as np
import mdtraj as md
from Bio import PDB
from scipy.spatial import KDTree
from collections import defaultdict
import gemmi
import os
from tqdm import tqdm
import pandas as pd
import itertools

# calculate averages for statistics
def calculate_averages(statistics, handle_aa=False):
    # Initialize a dictionary to hold the sums of each statistic
    sums = {}
    # Initialize a dictionary to hold the sums of each amino acid count
    aa_sums = {}

    # Iterate over the model numbers
    for model_num in range(5):  # assumes models are numbered 0 through 4
        # Check if the model's data exists
        if model_num in statistics:
            # Get the model's statistics
            model_stats = statistics[model_num]
            # For each statistic, add its value to the sum
            for stat, value in model_stats.items():
                # If this is the first time we've seen this statistic, initialize its sum to 0
                if stat not in sums:
                    sums[stat] = 0

                if value is None:
                    value = 0

                # If the statistic is mpnn_interface_AA and we're supposed to handle it separately, do so
                if handle_aa and stat == 'InterfaceAAs':
                    for aa, count in value.items():
                        # If this is the first time we've seen this amino acid, initialize its sum to 0
                        if aa not in aa_sums:
                            aa_sums[aa] = 0
                        aa_sums[aa] += count
                else:
                    sums[stat] += value

    # Now that we have the sums, we can calculate the averages
    averages = {stat: round(total / len(statistics), 2) for stat, total in sums.items()}

    # If we're handling aa counts, calculate their averages
    if handle_aa:
        aa_averages = {aa: round(total / len(statistics),2) for aa, total in aa_sums.items()}
        averages['InterfaceAAs'] = aa_averages

    return averages



# https://gitlab.com/ElofssonLab/FoldDock
def parse_gemmi_atom(residue, chain, atom):
    '''Parse atom information using gemmi Residue and Atom objects'''
    record = defaultdict()
    record['name'] = 'ATOM'
    record['atm_no'] = atom.serial  # Atom serial number
    record['atm_name'] = atom.name.strip()  # Atom name
    record['atm_alt'] = atom.altloc  # Alternate location indicator
    record['res_name'] = residue.name  # Residue name
    record['chain'] = chain.name  # Chain identifier
    record['res_no'] = residue.seqid.num  # Residue sequence number
    record['insert'] = residue.seqid.icode.strip()  # Insertion code
    record['resid'] = f"{residue.seqid.num}{residue.seqid.icode.strip()}"  # Unique residue identifier
    record['x'] = atom.pos.x  # X-coordinate
    record['y'] = atom.pos.y  # Y-coordinate
    record['z'] = atom.pos.z  # Z-coordinate
    record['occ'] = atom.occ  # Occupancy
    record['B'] = atom.b_iso  # B-factor

    return record

def parse_structure(structure):
    '''parse structure'''
    chain_coords, chain_plddt = {}, {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Pass chain name explicitly to the function
                    record = parse_gemmi_atom(residue, chain, atom)
                    # Get CB - CA for GLY
                    if record['atm_name'] == 'CB' or (record['atm_name'] == 'CA' and record['res_name'] == 'GLY'):
                        if record['chain'] in chain_coords:
                            chain_coords[record['chain']].append([record['x'], record['y'], record['z']])
                            chain_plddt[record['chain']].append(record['B'])
                        else:
                            chain_coords[record['chain']] = [[record['x'], record['y'], record['z']]]
                            chain_plddt[record['chain']] = [record['B']]
                            
    # Convert to arrays
    for chain in chain_coords:
        chain_coords[chain] = np.array(chain_coords[chain])
        chain_plddt[chain] = np.array(chain_plddt[chain])

    return chain_coords, chain_plddt


def score_complex(path_coords, path_CB_inds, path_plddt):
    '''Score all interfaces in the current complex
    '''
    metrics = {'Chain':[], 'n_ints':[], 'sum_av_IF_plDDT':[],
                'n_contacts':[], 'n_IF_residues':[]}

    chains = [*path_coords.keys()]
    chain_inds = np.arange(len(chains))
    #Get interfaces per chain
    for i in chain_inds:
        chain_i = chains[i]
        chain_coords = np.array(path_coords[chain_i])
        chain_CB_inds = path_CB_inds[chain_i]
        l1 = len(chain_CB_inds)
        chain_CB_coords = chain_coords[chain_CB_inds]
        chain_plddt = np.array(path_plddt[chain_i])
        #Metrics
        n_chain_ints = 0
        chain_av_IF_plDDT = 0
        n_chain_contacts = 0
        n_chain_IF_residues = 0

        for int_i in np.setdiff1d(chain_inds, i):
            int_chain = chains[int_i]
            int_chain_CB_coords = np.array(path_coords[int_chain])[path_CB_inds[int_chain]]
            int_chain_plddt = np.array(path_plddt[int_chain])
            #Calc 2-norm
            mat = np.append(chain_CB_coords,int_chain_CB_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            contact_dists = dists[:l1,l1:]
            contacts = np.argwhere(contact_dists<=8)
            #The first axis contains the contacts from chain 1
            #The second the contacts from chain 2
            if contacts.shape[0]>0:
                n_chain_ints += 1
                chain_av_IF_plDDT +=  np.concatenate((chain_plddt[contacts[:,0]], int_chain_plddt[contacts[:,1]])).mean()
                n_chain_contacts += contacts.shape[0]
                n_chain_IF_residues += np.unique(contacts).shape[0]

        #Save
        metrics['Chain'].append(chain_i)
        metrics['n_ints'].append(n_chain_ints)
        metrics['sum_av_IF_plDDT'].append(chain_av_IF_plDDT) #Divide with n_ints to get avg per int
        metrics['n_contacts'].append(n_chain_contacts)
        metrics['n_IF_residues'].append(n_chain_IF_residues)
    #Create df
    metrics_df = pd.DataFrame.from_dict(metrics)
    return metrics_df

def calc_mpDockQ(metrics_df):
    # https://www.nature.com/articles/s41467-022-33729-4
    '''Calculats the multiple interface pDockQ
    '''

    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0)))+b
        return y


    av_IF_plDDT = np.average(metrics_df.sum_av_IF_plDDT/metrics_df.n_ints)
    n_contacts= metrics_df.n_contacts.sum()

    L = 0.783
    x0= 289.79
    k= 0.061
    b= 0.23
    mpDockQ = sigmoid(av_IF_plDDT*np.log10(n_contacts+0.001), L ,x0, k, b)

    return mpDockQ

# Distance threshold, set to 8 Å
def calc_pdockq(structure, t = 8, ch1='A', ch2='B'):
    '''Calculate the pDockQ scores
    pdockQ = L / (1 + np.exp(-k*(x-x0)))+b
    L= 0.724 x0= 152.611 k= 0.052 and b= 0.018
    '''
    chain_coords, chain_plddt = parse_structure(structure)
    coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    # Calc 2-norm
    mat = np.append(coords1, coords2, axis=0)
    a_min_b = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1, l1:]  # upper triangular --> first dim = chain 1
    contacts = np.argwhere(contact_dists <= t)

    if contacts.shape[0] < 1:
        pdockq = 0
        ppv = 0
        n_if_contacts = 0
    else:
        # Get the average interface plDDT
        avg_if_plddt = np.average(np.concatenate([plddt1[np.unique(contacts[:, 0])], plddt2[np.unique(contacts[:, 1])]]))
        # Get the number of interface contacts
        n_if_contacts = contacts.shape[0]
        x = avg_if_plddt * np.log10(n_if_contacts)
        pdockq = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

        # PPV
        PPV = np.array([0.98128027, 0.96322524, 0.95333044, 0.9400192 ,
            0.93172991, 0.92420274, 0.91629946, 0.90952562, 0.90043139,
            0.8919553 , 0.88570037, 0.87822061, 0.87116417, 0.86040801,
            0.85453785, 0.84294946, 0.83367787, 0.82238224, 0.81190228,
            0.80223507, 0.78549007, 0.77766077, 0.75941223, 0.74006263,
            0.73044282, 0.71391784, 0.70615739, 0.68635536, 0.66728511,
            0.63555449, 0.55890174])

        pdockq_thresholds = np.array([0.67333079, 0.65666073, 0.63254566, 0.62604391,
            0.60150931, 0.58313803, 0.5647381 , 0.54122438, 0.52314392,
            0.49659878, 0.4774676 , 0.44661346, 0.42628389, 0.39990988,
            0.38479715, 0.3649393 , 0.34526004, 0.3262589 , 0.31475668,
            0.29750023, 0.26673725, 0.24561247, 0.21882689, 0.19651314,
            0.17606258, 0.15398168, 0.13927677, 0.12024131, 0.09996019,
            0.06968505, 0.02946438])

        inds = np.argwhere(pdockq_thresholds>=pdockq)
        if len(inds)>0:
            ppv = PPV[inds[-1]][0]
        else:
            ppv = PPV[0]

    return pdockq, ppv, n_if_contacts


def extract_interface_atoms(pdb_file, chain1="A", chain2="B", cutoff=4.0):
    """ Extract atoms within a given cutoff distance at the interface of two chains. """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Store atom coordinates
    chain_atoms = {chain1: [], chain2: []}
    
    for model in structure:
        for chain in model:
            if chain.id in [chain1, chain2]:
                for residue in chain:
                    if residue.get_id()[0] == " ":  # Ignore heteroatoms
                        for atom in residue:
                            chain_atoms[chain.id].append(atom.coord)

    # Convert to NumPy arrays
    points1 = np.array(chain_atoms[chain1])
    points2 = np.array(chain_atoms[chain2])

    return points1, points2

def compute_shape_complementarity(pdb_file, chain1="A", chain2="B", cutoff=4.0):
    """ Compute shape complementarity score (SC) based on interface distance. """
    points1, points2 = extract_interface_atoms(pdb_file, chain1, chain2, cutoff)

    if len(points1) == 0 or len(points2) == 0:
        return 0.0  # No interface detected

    # Build KD-Trees
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)

    # Compute distances from chain1 to chain2
    distances1, _ = tree1.query(points2, k=1)
    distances2, _ = tree2.query(points1, k=1)

    # Compute mean distance (lower = better complementarity)
    mean_dist = np.mean(np.concatenate([distances1, distances2]))

    # Normalize complementarity score (higher = better)
    sc_score = np.exp(-mean_dist)  # Exponential decay: SC ~ [0,1]

    return sc_score

# Function 5: Number of unsaturated hydrogen bonds at interface (< 3)
def compute_unsaturated_hbonds(pdb_file):
    traj = md.load(pdb_file)
    hbonds = md.baker_hubbard(traj, periodic=False)
    unsaturated = len(hbonds)
    return unsaturated

# Function 6: Hydrophobicity of binder surface (< 35%)
def compute_hydrophobicity(structure):
    hydrophobic_residues = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO']
    total_residues = 0
    hydrophobic_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                total_residues += 1
                if residue.name in hydrophobic_residues:
                    hydrophobic_count += 1
                    
    return (hydrophobic_count / total_residues) * 100
