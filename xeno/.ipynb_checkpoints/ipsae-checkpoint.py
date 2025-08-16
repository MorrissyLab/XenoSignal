import sys
import os
import math
import json
import numpy as np
from typing import Dict, List, Any, Set, Tuple

# Configure numpy printing (useful for debugging, can be removed for production)
# np.set_printoptions(threshold=np.inf)

# --- Helper Functions (can be defined outside the main function) ---

def ptm_func(x: float, d0: float) -> float:
    """Calculates the Predicted Temaplate Modeling score component."""
    return 1.0 / (1 + (x / d0) ** 2.0)

# Vectorized version of ptm_func for NumPy arrays
ptm_func_vec = np.vectorize(ptm_func)

def calc_d0(L: float, pair_type: str) -> float:
    """
    Calculates the d0 parameter for the ptm function based on sequence length L.
    Based on Yang and Skolnick, PROTEINS: Structure, Function, and Bioinformatics 57:702–710 (2004).
    Adjusted minimum value for nucleic acids.
    """
    L = float(L)
    if L < 27:
        L = 27
    min_value = 1.0
    if pair_type == 'nucleic_acid':
        min_value = 2.0 # Arbitrary minimum for NA
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)

def calc_d0_array(L: np.ndarray, pair_type: str) -> np.ndarray:
    """Vectorized calculation of d0 for a NumPy array of lengths L."""
    L = np.maximum(27.0, L.astype(float))
    min_value = 1.0
    if pair_type == 'nucleic_acid':
        min_value = 2.0 # Arbitrary minimum for NA
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)

def parse_pdb_atom_line(line: str) -> Dict[str, Any] | None:
    """Parses a PDB ATOM or HETATM line."""
    if len(line) < 54: # Minimum length for coordinates
        return None # Not a valid atom line with coordinates
    try:
        atom_num = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21].strip()
        # PDB residue numbers can have insertion codes, handle them
        residue_seq_num_str = line[22:26].strip()
        insert_code = line[26].strip()
        residue_seq_num = int(residue_seq_num_str)
        
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        
        return {
            'record_name': line[0:6].strip(),
            'atom_num': atom_num,
            'atom_name': atom_name,
            'residue_name': residue_name,
            'chain_id': chain_id,
            'residue_seq_num': residue_seq_num,
            'insert_code': insert_code if insert_code else None, # Store insert code if present
            'x': x,
            'y': y,
            'z': z,
            'coor': np.array([x, y, z])
        }
    except (ValueError, IndexError):
        return None # Failed to parse the line

def parse_cif_atom_line(line: str, fielddict: Dict[str, int]) -> Dict[str, Any] | None:
    """Parses an mmCIF ATOM or HETATM line using a field dictionary."""
    linelist = line.split()
    
    # Check if essential fields exist and the line is long enough
    essential_fields = ['id', 'label_atom_id', 'label_comp_id', 'label_asym_id', 'label_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z']
    if not all(field in fielddict for field in essential_fields) or any(fielddict[field] >= len(linelist) for field in essential_fields):
        return None # Missing essential fields or line is too short

    try:
        atom_num_str = linelist[fielddict['id']]
        atom_name = linelist[fielddict['label_atom_id']]
        residue_name = linelist[fielddict['label_comp_id']]
        chain_id = linelist[fielddict['label_asym_id']]
        residue_seq_num_str = linelist[fielddict['label_seq_id']]
        
        x_str = linelist[fielddict['Cartn_x']]
        y_str = linelist[fielddict['Cartn_y']]
        z_str = linelist[fielddict['Cartn_z']]

        # Check for ligand atoms indicated by '.' in label_seq_id
        if residue_seq_num_str == ".":
             # Still return data for ligands but mark them as such
             return {
                 'record_name': linelist[0], # ATOM or HETATM
                 'atom_num': int(atom_num_str),
                 'atom_name': atom_name,
                 'residue_name': residue_name,
                 'chain_id': chain_id,
                 'residue_seq_num': None, # No sequence number for ligand
                 'insert_code': None,
                 'x': float(x_str),
                 'y': float(y_str),
                 'z': float(z_str),
                 'coor': np.array([float(x_str), float(y_str), float(z_str)]),
                 'is_ligand': True # Flag for ligands
             }
        
        # Parse as standard residue
        atom_num = int(atom_num_str)
        residue_seq_num = int(residue_seq_num_str)
        x = float(x_str)
        y = float(y_str)
        z = float(z_str)

        # Check for insertion code if available in fielddict
        insert_code = None
        if 'pdbx_PDB_ins_code' in fielddict and fielddict['pdbx_PDB_ins_code'] < len(linelist):
             ins_code_val = linelist[fielddict['pdbx_PDB_ins_code']]
             if ins_code_val != '.':
                  insert_code = ins_code_val
        
        return {
            'record_name': linelist[0], # ATOM or HETATM
            'atom_num': atom_num,
            'atom_name': atom_name,
            'residue_name': residue_name,
            'chain_id': chain_id,
            'residue_seq_num': residue_seq_num,
            'insert_code': insert_code,
            'x': x,
            'y': y,
            'z': z,
            'coor': np.array([x, y, z]),
            'is_ligand': False # Flag for ligands
        }
    except (ValueError, IndexError):
        return None # Failed to parse the line

def contiguous_ranges(numbers: Set[int] | List[int]) -> str:
    """Converts a set or list of residue numbers into a PyMOL-compatible range string."""
    if not numbers:
        return ""
    
    sorted_numbers = sorted(list(numbers)) # Convert set to list and sort
    start = sorted_numbers[0]
    end = start
    ranges = []
    
    def format_range(start_val, end_val):
        if start_val == end_val:
            return f"{start_val}"
        else:
            return f"{start_val}-{end_val}"
            
    for number in sorted_numbers[1:]:
        if number == end + 1:
            end = number
        else:
            ranges.append(format_range(start, end))
            start = end = number
            
    ranges.append(format_range(start, end)) # Append the last range
    return '+'.join(ranges)

def init_chainpairdict_zeros(chainlist: List[str]) -> Dict[str, Dict[str, float]]:
    """Initializes a nested dictionary for chain pairs with values set to 0.0."""
    return {chain1: {chain2: 0.0 for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

def init_chainpairdict_npzeros(chainlist: List[str], arraysize: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Initializes a nested dictionary for chain pairs with NumPy arrays of zeros."""
    return {chain1: {chain2: np.zeros(arraysize) for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

def init_chainpairdict_set(chainlist: List[str]) -> Dict[str, Dict[str, Set[int]]]:
    """Initializes a nested dictionary for chain pairs with empty sets."""
    return {chain1: {chain2: set() for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

def classify_chains(chains: np.ndarray, residue_types: np.ndarray) -> Dict[str, str]:
    """Classifies chains as 'protein' or 'nucleic_acid'."""
    nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    chain_types = {}
    
    unique_chains = np.unique(chains)
    for chain in unique_chains:
        indices = np.where(chains == chain)[0]
        chain_residues = residue_types[indices]
        
        # Check if any residue in the chain is a known nucleic acid residue
        is_nucleic_acid = any(res in nuc_residue_set for res in chain_residues)
        
        chain_types[chain] = 'nucleic_acid' if is_nucleic_acid else 'protein'
    
    return chain_types

# --- Main Function ---

def calculate_ipsae_scores(
    pae_file_path: str,
    structure_file_path: str,
    pae_cutoff: float,
    dist_cutoff: float
) -> Dict[str, Any]:
    """
    Calculates ipSAE and other scores (pDockQ, pDockQ2, LIS) for pairwise
    protein-protein (and protein-nucleic acid/nucleic acid-nucleic acid)
    interactions from AlphaFold2/3 or Boltz1 output files.

    Args:
        pae_file_path: Path to the PAE JSON (AF2/AF3) or NPZ (Boltz1) file.
        structure_file_path: Path to the PDB (AF2) or mmCIF (AF3/Boltz1) file.
        pae_cutoff: The PAE cutoff value (in Angstroms) to use for ipSAE.
        dist_cutoff: The distance cutoff value (in Angstroms) for identifying
                     interface residues for visualization and counting.

    Returns:
        A dictionary containing calculated scores and data for potential output
        files.

    Raises:
        FileNotFoundError: If the input files do not exist.
        ValueError: If the input file types are not supported or parsing fails.
    """
    
    # --- Determine file types and setup ---
    
    af2, af3, boltz1, cif = False, False, False, False
    
    if structure_file_path.lower().endswith(".pdb"):
        if pae_file_path.lower().endswith(('.json', '.pkl')):
            af2 = True
        else:
             raise ValueError(f"Mismatch: PDB file ({structure_file_path}) requires a .json or .pkl PAE file, but got {pae_file_path}")
    elif structure_file_path.lower().endswith(".cif"):
        cif = True
        if pae_file_path.lower().endswith(".json"):
            af3 = True
        elif pae_file_path.lower().endswith(".npz"):
            boltz1 = True
        else:
            raise ValueError(f"Mismatch: mmCIF file ({structure_file_path}) requires a .json (AF3) or .npz (Boltz1) PAE file, but got {pae_file_path}")
    else:
        raise ValueError(f"Unsupported structure file type: {structure_file_path}. Please provide a .pdb or .cif file.")

    if not os.path.exists(pae_file_path):
        raise FileNotFoundError(f"PAE file not found: {pae_file_path}")
    if not os.path.exists(structure_file_path):
        raise FileNotFoundError(f"Structure file not found: {structure_file_path}")

    # Base name for potential output files (without extension)
    path_stem = os.path.splitext(structure_file_path)[0]
    pae_string = f"{int(pae_cutoff):02}" # Format with leading zero if < 10
    dist_string = f"{int(dist_cutoff):02}" # Format with leading zero if < 10
    output_stem = f'{path_stem}_{pae_string}_{dist_string}'
    
    # --- Parse Structure File ---
    
    residues_data = []
    cb_residues_data = [] # Using CB for proteins, C3' for nucleic acids for pDockQ/pDockQ2
    chains = []
    residue_identifiers = [] # e.g., "ALA A 15"
    
    # For AF3 and Boltz1 CIFs: need mapping from atom_site field name to index
    atomsitefield_dict = {}
    
    # Amino acid and Nucleic acid 3-letter codes for identifying residues
    protein_residue_set = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                           "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
    nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    main_chain_atoms = {"CA", "C1'"} # Using C1' for nucleic acids

    # Token mask for mapping full PAE matrix (from json/npz) to residues
    # 1 if the atom is a representative residue atom (CA for protein, C1' for NA)
    # 0 otherwise (ligands, solvent, other atoms in modified residues, etc.)
    token_mask = []
    
    # List to store all parsed atoms, useful for later indexing
    all_parsed_atoms = []

    with open(structure_file_path, 'r') as structure_file:
        for line in structure_file:
            line = line.strip()
            if not line: continue # Skip empty lines

            if cif and line.startswith("_atom_site."):
                # Build the field dictionary for mmCIF files
                try:
                    atomsite, fieldname = line.split(".", 1) # Split only on the first '.'
                    atomsitefield_dict[fieldname] = len(atomsitefield_dict)
                except ValueError:
                    print(f"Warning: Could not parse mmCIF header line: {line}")
                    continue # Skip malformed lines

            if line.startswith("ATOM") or line.startswith("HETATM"):
                if cif:
                    atom = parse_cif_atom_line(line, atomsitefield_dict)
                else: # PDB
                    atom = parse_pdb_atom_line(line)

                if atom is None:
                    print(f"Warning: Could not parse atom line: {line}")
                    continue

                all_parsed_atoms.append(atom)
                
                # Determine if this atom represents a residue token
                is_residue_token = False
                
                if not atom.get('is_ligand', False) and atom['residue_name'] in (protein_residue_set | nuc_residue_set):
                     # It's a standard protein or nucleic acid residue
                     if atom['atom_name'] in main_chain_atoms:
                          is_residue_token = True
                          # Store data for the main residue list (used for PAE indexing)
                          residues_data.append({
                              'atom_num': atom['atom_num'],
                              'coor': atom['coor'], # Using CA/C1' for main coordinate list
                              'res': atom['residue_name'],
                              'chainid': atom['chain_id'],
                              'resnum': atom['residue_seq_num'],
                              'insert': atom['insert_code'],
                              'residue_id': f"{atom['residue_name']:3s} {atom['chain_id']:3s} {atom['residue_seq_num']}{atom['insert'] if atom['insert'] else '':1s}".strip()
                          })
                          chains.append(atom['chain_id'])
                          residue_identifiers.append(residues_data[-1]['residue_id'])

                     # Store data for the coordinate list used for distance calculations (CB or C3')
                     if (atom['residue_name'] in protein_residue_set and atom['atom_name'] == "CB") or \
                        (atom['residue_name'] == "GLY" and atom['atom_name'] == "CA") or \
                        (atom['residue_name'] in nuc_residue_set and atom['atom_name'] == "C3'"):
                          cb_residues_data.append({
                              'atom_num': atom['atom_num'],
                              'coor': atom['coor'], # Using CB/C3' for distance calculations
                              'res': atom['residue_name'],
                              'chainid': atom['chain_id'],
                              'resnum': atom['residue_seq_num'],
                              'insert': atom['insert_code'],
                              'residue_id': f"{atom['residue_name']:3s} {atom['chain_id']:3s} {atom['residue_seq_num']}{atom['insert'] if atom['insert'] else '':1s}".strip()
                          })
                # Add to token mask: 1 if it's a residue token (CA/C1'), 0 otherwise
                token_mask.append(1 if is_residue_token else 0)


    # Check if any residues were parsed
    if not residues_data:
         raise ValueError("No valid protein or nucleic acid residues found in the structure file.")

    # Convert structure information lists to numpy arrays
    num_residues = len(residues_data) # Number of residues used for PAE/pLDDT indexing
    num_cb_residues = len(cb_residues_data) # Number of residues used for distance/pDockQ indexing

    if num_residues == 0 or num_cb_residues == 0:
         raise ValueError("Could not extract necessary atom coordinates (CA/C1' and CB/C3').")
         
    # Ensure the number of residues for PAE and distance calculations match
    # This is important because PAE is defined per-residue, and we use CB/C3'
    # for distance, but the PAE matrix corresponds to the main residue list.
    # If they don't match, something is wrong with the parsing or input file.
    if num_residues != num_cb_residues:
         # This could happen if some residues are missing CA/C1' or CB/C3'
         # For now, we'll raise an error, but a more robust solution might
         # try to match residues based on chain/seq_num/insert.
         print("Warning: Number of CA/C1' residues does not match number of CB/C3' residues. Distance calculations may be inaccurate.")
         # Optionally, proceed but be aware of the issue, or raise an error:
         # raise ValueError(f"Mismatch between number of CA/C1' residues ({num_residues}) and CB/C3' residues ({num_cb_residues}).")

    # Create arrays based on the residues_data list (used for PAE indexing)
    CA_atom_num_for_token = np.array([res['atom_num'] - 1 for res in residues_data]) # Use 0-based index
    chains_np = np.array([res['chainid'] for res in residues_data])
    unique_chains = sorted(list(np.unique(chains_np))) # Sorted list of unique chains
    residue_types_np = np.array([res['res'] for res in residues_data])
    residue_ids_np = np.array([res['residue_id'] for res in residues_data])

    # Create coordinates array based on cb_residues_data list (used for distance)
    coordinates_np = np.array([res['coor'] for res in cb_residues_data])

    token_array = np.array(token_mask, dtype=bool) # Boolean mask for token filtering
    num_tokens = np.sum(token_array) # Number of tokens (should equal num_residues if parsing worked)

    if num_tokens != num_residues:
        print(f"Warning: Number of selected residue tokens ({num_tokens}) does not match the number of parsed CA/C1' residues ({num_residues}). This might indicate issues with the token mask or input file.")
        # We will proceed using the PAE/pLDDT data corresponding to the selected tokens.
        # The size of the filtered PAE/pLDDT arrays will be num_tokens x num_tokens or num_tokens.

    # Classify chains as protein or nucleic acid
    chain_dict = classify_chains(chains_np, residue_types_np)

    # Determine chain pair types for d0 calculation
    chain_pair_type = init_chainpairdict_zeros(unique_chains)
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2: continue
            if chain_dict[chain1] == 'nucleic_acid' or chain_dict[chain2] == 'nucleic_acid':
                chain_pair_type[chain1][chain2] = 'nucleic_acid'
            else:
                chain_pair_type[chain1][chain2] = 'protein'

    # Calculate distance matrix using NumPy broadcasting on CB/C3' coordinates
    if num_cb_residues > 0:
       distances_np = np.sqrt(((coordinates_np[:, np.newaxis, :] - coordinates_np[np.newaxis, :, :])**2).sum(axis=2))
    else:
       distances_np = np.zeros((num_residues, num_residues)) # Fallback, though should not happen if parsing works

    # --- Load PAE and pLDDT Data ---

    plddt = None
    pae_matrix = None
    chain_pair_iptm_data = None # For AF3/Boltz1 chain_pair_iptm

    if af2:
        # AF2 JSON/PKL structure: {'plddt': [], 'pae': [[]], 'iptm': float, 'ptm': float}
        try:
            if pae_file_path.lower().endswith('.pkl'):
                 # Assuming PKL contains similar structure to JSON
                 import pickle
                 with open(pae_file_path, 'rb') as f:
                      data = pickle.load(f)
            else: # JSON
                with open(pae_file_path, 'r') as f:
                    data = json.load(f)

            if 'plddt' in data:
                 # AF2 plddt is per-residue
                plddt = np.array(data['plddt'])
                # For pDockQ/pDockQ2, use the same pLDDT as it's per-residue
                cb_plddt = plddt.copy() 
            else:
                print("Warning: 'plddt' not found in AF2 PAE file. Setting pLDDT to zeros.")
                plddt = np.zeros(num_residues)
                cb_plddt = np.zeros(num_residues) # Use same size zeros
                
            if 'pae' in data:
                 # AF2 pae is already num_residues x num_residues
                 pae_matrix = np.array(data['pae'])
            elif 'predicted_aligned_error' in data:
                 pae_matrix = np.array(data['predicted_aligned_error'])
            else:
                 raise ValueError("'pae' or 'predicted_aligned_error' not found in AF2 PAE file.")

            # AF2 might have overall iptm/ptm scores in the JSON/PKL
            # We won't use these directly for chain-specific calculations but can return them
            overall_iptm_af2 = data.get('iptm', -1.0)
            overall_ptm_af2 = data.get('ptm', -1.0)
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError, ImportError, EOFError) as e:
            raise ValueError(f"Error loading/parsing AF2 PAE file {pae_file_path}: {e}")

    elif boltz1:
        # Boltz1 NPZ structure: {'pae': np.ndarray, 'plddt': np.ndarray (0-1 scale)}
        # Boltz1 JSON structure: {'pair_chains_iptm': {}, ...}
        
        # Load pLDDT from NPZ
        plddt_file_path = pae_file_path.replace("pae_", "plddt_").replace(".npz", ".npz") # Standard naming
        if not os.path.exists(plddt_file_path):
             # Try the same directory as the structure file
             plddt_file_path_alt = os.path.join(os.path.dirname(structure_file_path), os.path.basename(plddt_file_path))
             if not os.path.exists(plddt_file_path_alt):
                  print(f"Warning: Boltz1 pLDDT file not found at {plddt_file_path} or {plddt_file_path_alt}. Setting pLDDT to zeros.")
                  plddt_boltz1_full = np.zeros(num_tokens) # Use num_tokens size
             else:
                  plddt_file_path = plddt_file_path_alt # Use the alternative path
                  try:
                       data_plddt = np.load(plddt_file_path)
                       # Boltz1 plddt is per-atom, on 0-1 scale, need to scale to 0-100 and filter by token mask
                       plddt_boltz1_full = np.array(100.0 * data_plddt['plddt'])
                  except (IOError, KeyError) as e:
                       print(f"Warning: Error loading/parsing Boltz1 pLDDT NPZ file {plddt_file_path}: {e}. Setting pLDDT to zeros.")
                       plddt_boltz1_full = np.zeros(num_tokens) # Use num_tokens size
        else:
             try:
                  data_plddt = np.load(plddt_file_path)
                  # Boltz1 plddt is per-atom, on 0-1 scale, need to scale to 0-100 and filter by token mask
                  plddt_boltz1_full = np.array(100.0 * data_plddt['plddt'])
             except (IOError, KeyError) as e:
                  print(f"Warning: Error loading/parsing Boltz1 pLDDT NPZ file {plddt_file_path}: {e}. Setting pLDDT to zeros.")
                  plddt_boltz1_full = np.zeros(num_tokens) # Use num_tokens size

        # Apply token mask to get per-residue plddt for tokens
        if len(plddt_boltz1_full) != len(token_array):
             print(f"Warning: Full Boltz1 pLDDT array size ({len(plddt_boltz1_full)}) does not match token mask size ({len(token_array)}). Cannot apply mask.")
             # Attempt to use it directly, assuming it's already filtered (less likely) or just pad/truncate
             # A safer approach might be to try mapping atoms, but for now, issue warning.
             # If sizes don't match, filtering will fail. Let's just use zeros.
             plddt = np.zeros(num_tokens)
             cb_plddt = np.zeros(num_tokens)
        else:
             plddt = plddt_boltz1_full[token_array]
             # For pDockQ/pDockQ2, use the same filtered pLDDT
             cb_plddt = plddt.copy()


        # Load PAE from NPZ
        try:
            data_pae = np.load(pae_file_path)
            pae_matrix_boltz1_full = np.array(data_pae['pae'])
            
            # Apply token mask to get per-token PAE matrix
            if pae_matrix_boltz1_full.shape[0] != len(token_array) or pae_matrix_boltz1_full.shape[1] != len(token_array):
                 print(f"Warning: Full Boltz1 PAE matrix shape {pae_matrix_boltz1_full.shape} does not match token mask size {len(token_array)}. Cannot apply mask.")
                 # Use a zeros matrix of the expected filtered size
                 pae_matrix = np.zeros((num_tokens, num_tokens))
            else:
                 pae_matrix = pae_matrix_boltz1_full[np.ix_(token_array, token_array)]

        except (IOError, KeyError) as e:
            raise ValueError(f"Error loading/parsing Boltz1 PAE NPZ file {pae_file_path}: {e}")

        # Load chain_pair_iptm from JSON summary
        summary_file_path = pae_file_path.replace("pae_", "confidence_").replace(".npz", ".json") # Standard naming
        if not os.path.exists(summary_file_path):
              # Try the same directory as the structure file
              summary_file_path_alt = os.path.join(os.path.dirname(structure_file_path), os.path.basename(summary_file_path))
              if not os.path.exists(summary_file_path_alt):
                   print(f"Warning: Boltz1 summary file not found at {summary_file_path} or {summary_file_path_alt}. Chain pair iptm will be unavailable.")
                   chain_pair_iptm_data = {}
              else:
                   summary_file_path = summary_file_path_alt
                   try:
                       with open(summary_file_path, 'r') as f:
                           data_summary = json.load(f)
                           # Boltz1 stores chain pair iptm by index (0, 1, 2...)
                           if 'pair_chains_iptm' in data_summary:
                               chain_pair_iptm_data_raw = data_summary['pair_chains_iptm']
                               chain_pair_iptm_data = {}
                               # Map index keys to chain IDs (assuming A=0, B=1, ...)
                               for c1_idx_str, values in chain_pair_iptm_data_raw.items():
                                   try:
                                       c1_idx = int(c1_idx_str)
                                       chain1 = chr(ord('A') + c1_idx)
                                       if chain1 in unique_chains:
                                            chain_pair_iptm_data[chain1] = {}
                                            for c2_idx_str, iptm_value in values.items():
                                                try:
                                                    c2_idx = int(c2_idx_str)
                                                    chain2 = chr(ord('A') + c2_idx)
                                                    if chain2 in unique_chains and chain1 != chain2:
                                                        chain_pair_iptm_data[chain1][chain2] = iptm_value
                                                except ValueError:
                                                     print(f"Warning: Could not parse chain index '{c2_idx_str}' in Boltz1 summary.")
                                   except ValueError:
                                        print(f"Warning: Could not parse chain index '{c1_idx_str}' in Boltz1 summary.")
                           else:
                               print("Warning: 'pair_chains_iptm' not found in Boltz1 summary file. Chain pair iptm will be unavailable.")
                               chain_pair_iptm_data = {}

                   except (json.JSONDecodeError, KeyError) as e:
                       print(f"Warning: Error loading/parsing Boltz1 summary file {summary_file_path}: {e}. Chain pair iptm will be unavailable.")
                       chain_pair_iptm_data = {}
        else:
              try:
                  with open(summary_file_path, 'r') as f:
                      data_summary = json.load(f)
                      # Boltz1 stores chain pair iptm by index (0, 1, 2...)
                      if 'pair_chains_iptm' in data_summary:
                          chain_pair_iptm_data_raw = data_summary['pair_chains_iptm']
                          chain_pair_iptm_data = {}
                          # Map index keys to chain IDs (assuming A=0, B=1, ...)
                          for c1_idx_str, values in chain_pair_iptm_data_raw.items():
                              try:
                                  c1_idx = int(c1_idx_str)
                                  chain1 = chr(ord('A') + c1_idx)
                                  if chain1 in unique_chains:
                                       chain_pair_iptm_data[chain1] = {}
                                       for c2_idx_str, iptm_value in values.items():
                                           try:
                                               c2_idx = int(c2_idx_str)
                                               chain2 = chr(ord('A') + c2_idx)
                                               if chain2 in unique_chains and chain1 != chain2:
                                                   chain_pair_iptm_data[chain1][chain2] = iptm_value
                                           except ValueError:
                                                print(f"Warning: Could not parse chain index '{c2_idx_str}' in Boltz1 summary.")
                              except ValueError:
                                   print(f"Warning: Could not parse chain index '{c1_idx_str}' in Boltz1 summary.")
                      else:
                          print("Warning: 'pair_chains_iptm' not found in Boltz1 summary file. Chain pair iptm will be unavailable.")
                          chain_pair_iptm_data = {}
              except (json.JSONDecodeError, KeyError) as e:
                  print(f"Warning: Error loading/parsing Boltz1 summary file {summary_file_path}: {e}. Chain pair iptm will be unavailable.")
                  chain_pair_iptm_data = {}

        # Check if PAE matrix was loaded and filtered correctly
        if pae_matrix is None or pae_matrix.shape != (num_tokens, num_tokens):
             raise ValueError(f"Boltz1 PAE matrix could not be loaded or filtered to the expected size ({num_tokens}x{num_tokens}).")

        # Check if pLDDT was loaded and filtered correctly
        if plddt is None or len(plddt) != num_tokens:
             raise ValueError(f"Boltz1 pLDDT could not be loaded or filtered to the expected size ({num_tokens}).")
             
    elif af3:
        # AF3 JSON structure: {'atom_plddts': [], 'pae': [[]], ...}
        # AF3 Summary JSON structure: {'chain_pair_iptm': [[]], ...}
        
        try:
            with open(pae_file_path, 'r') as f:
                data = json.load(f)

            if 'atom_plddts' in data:
                 # AF3 plddt is per-atom, need to filter by token mask
                atom_plddts_full = np.array(data['atom_plddts'])
                if len(atom_plddts_full) != len(token_array):
                     print(f"Warning: Full AF3 atom_plddts array size ({len(atom_plddts_full)}) does not match token mask size ({len(token_array)}). Cannot apply mask.")
                     plddt = np.zeros(num_tokens)
                     cb_plddt = np.zeros(num_tokens)
                else:
                     plddt = atom_plddts_full[token_array]
                     # For pDockQ/pDockQ2, use the same filtered pLDDT
                     cb_plddt = plddt.copy()
            else:
                print("Warning: 'atom_plddts' not found in AF3 PAE file. Setting pLDDT to zeros.")
                plddt = np.zeros(num_tokens)
                cb_plddt = np.zeros(num_tokens)

            if 'pae' in data:
                pae_matrix_af3_full = np.array(data['pae'])
                # Apply token mask to get per-token PAE matrix
                if pae_matrix_af3_full.shape[0] != len(token_array) or pae_matrix_af3_full.shape[1] != len(token_array):
                     print(f"Warning: Full AF3 PAE matrix shape {pae_matrix_af3_full.shape} does not match token mask size {len(token_array)}. Cannot apply mask.")
                     pae_matrix = np.zeros((num_tokens, num_tokens))
                else:
                     pae_matrix = pae_matrix_af3_full[np.ix_(token_array, token_array)]
            else:
                 raise ValueError("'pae' not found in AF3 PAE file.")

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error loading/parsing AF3 PAE file {pae_file_path}: {e}")

        # Load chain_pair_iptm from JSON summary
        summary_file_path = None
        if "confidences" in pae_file_path:
            summary_file_path = pae_file_path.replace("confidences", "summary_confidences")
        elif "full_data" in pae_file_path:
            summary_file_path = pae_file_path.replace("full_data", "summary_confidences")

        if summary_file_path is not None and os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r') as f:
                    data_summary = json.load(f)
                    # AF3 stores chain pair iptm by index (0, 1, 2...)
                    if 'chain_pair_iptm' in data_summary:
                         chain_pair_iptm_data_raw = data_summary['chain_pair_iptm']
                         chain_pair_iptm_data = {}
                         # Map index keys to chain IDs (assuming A=0, B=1, ...)
                         # Note: chain_pair_iptm_data_raw is a list of lists
                         for c1_idx, values in enumerate(chain_pair_iptm_data_raw):
                             chain1 = chr(ord('A') + c1_idx)
                             if chain1 in unique_chains:
                                  chain_pair_iptm_data[chain1] = {}
                                  for c2_idx, iptm_value in enumerate(values):
                                       chain2 = chr(ord('A') + c2_idx)
                                       if chain2 in unique_chains and chain1 != chain2:
                                            chain_pair_iptm_data[chain1][chain2] = iptm_value
                         # Ensure all unique chains are represented, even if they have no partners
                         for chain in unique_chains:
                              if chain not in chain_pair_iptm_data:
                                   chain_pair_iptm_data[chain] = {}
                                   
                    else:
                        print("Warning: 'chain_pair_iptm' not found in AF3 summary file. Chain pair iptm will be unavailable.")
                        chain_pair_iptm_data = {}

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error loading/parsing AF3 summary file {summary_file_path}: {e}. Chain pair iptm will be unavailable.")
                chain_pair_iptm_data = {}
        else:
            print(f"Warning: AF3 summary file not found at {summary_file_path}. Chain pair iptm will be unavailable.")
            chain_pair_iptm_data = {}

        # Check if PAE matrix was loaded and filtered correctly
        if pae_matrix is None or pae_matrix.shape != (num_tokens, num_tokens):
             raise ValueError(f"AF3 PAE matrix could not be loaded or filtered to the expected size ({num_tokens}x{num_tokens}).")

        # Check if pLDDT was loaded and filtered correctly
        if plddt is None or len(plddt) != num_tokens:
             raise ValueError(f"AF3 pLDDT could not be loaded or filtered to the expected size ({num_tokens}).")

    # Check if pae_matrix and plddt were successfully loaded for any type
    if pae_matrix is None or plddt is None:
        raise ValueError("Failed to load PAE or pLDDT data from the provided files.")

    # Ensure filtered PAE/pLDDT match the number of residues we identified as tokens
    # The score calculations will use these filtered arrays.
    if pae_matrix.shape != (num_tokens, num_tokens) or plddt.shape[0] != num_tokens:
         # This check was done during loading, but double-check here
         raise RuntimeError("Internal error: Filtered PAE/pLDDT dimensions do not match number of tokens.")
    if cb_plddt.shape[0] != num_residues:
         # This should match the number of residues identified from the structure file for distances
         print(f"Warning: CB/C3' pLDDT array size ({cb_plddt.shape[0]}) does not match number of residues from structure ({num_residues}).")
         # We will proceed, but pDockQ/pDockQ2 calculations might be affected.


    # --- Calculate Scores ---

    # Initialize dictionaries to store calculated scores
    # These will store values per chain pair (e.g., A-B)

    # By-residue scores (chain1 residue perspective)
    ipsae_d0chn_byres: Dict[str, Dict[str, np.ndarray]] = init_chainpairdict_npzeros(unique_chains, num_residues)
    ipsae_d0dom_byres: Dict[str, Dict[str, np.ndarray]] = init_chainpairdict_npzeros(unique_chains, num_residues)
    ipsae_d0res_byres: Dict[str, Dict[str, np.ndarray]] = init_chainpairdict_npzeros(unique_chains, num_residues)
    
    # Chain-pair aggregate scores (asymmetric: A->B)
    ipsae_d0chn_asym: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_asym: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_asym: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)

    # Chain-pair aggregate scores (symmetric: max(A->B, B->A))
    ipsae_d0chn_max: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_max: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_max: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)

    # Residue identifiers contributing to max scores
    ipsae_d0chn_asymres: Dict[str, Dict[str, str]] = init_chainpairdict_zeros(unique_chains) # Stores residue_id
    ipsae_d0dom_asymres: Dict[str, Dict[str, str]] = init_chainpairdict_zeros(unique_chains) # Stores residue_id
    ipsae_d0res_asymres: Dict[str, Dict[str, str]] = init_chainpairdict_zeros(unique_chains) # Stores residue_id
    ipsae_d0chn_maxres: Dict[str, Dict[str, str]] = init_chainpairdict_zeros(unique_chains) # Stores residue_id
    ipsae_d0dom_maxres: Dict[str, Dict[str, str]] = init_chainpairdict_zeros(unique_chains) # Stores residue_id
    ipsae_d0res_maxres: Dict[str, Dict[str, str]] = init_chainpairdict_zeros(unique_chains) # Stores residue_id

    # Counts used in ipSAE calculation
    n0chn: Dict[str, Dict[str, int]] = init_chainpairdict_zeros(unique_chains)
    n0dom: Dict[str, Dict[str, int]] = init_chainpairdict_zeros(unique_chains)
    n0res_byres: Dict[str, Dict[str, np.ndarray]] = init_chainpairdict_npzeros(unique_chains, num_residues) # Number of chain2 residues with PAE < cutoff for each chain1 residue

    # d0 values used in ipSAE calculation
    d0chn: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)
    d0dom: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)
    d0res_byres: Dict[str, Dict[str, np.ndarray]] = init_chainpairdict_npzeros(unique_chains, num_residues) # d0 based on n0res_byres

    # Interface residue tracking (based on PAE < cutoff)
    pae_interface_residues: Dict[str, Dict[str, Set[Tuple[str, int, Any]]]] = {c1: {c2: set() for c2 in unique_chains if c1 != c2} for c1 in unique_chains} # Store (chainid, resnum, insert)
    pae_valid_pair_counts: Dict[str, Dict[str, int]] = init_chainpairdict_zeros(unique_chains) # Count of pairs with PAE < cutoff

    # Interface residue tracking (based on Distance < cutoff)
    dist_interface_residues: Dict[str, Dict[str, Set[Tuple[str, int, Any]]]] = {c1: {c2: set() for c2 in unique_chains if c1 != c2} for c1 in unique_chains} # Store (chainid, resnum, insert)
    dist_valid_pair_counts: Dict[str, Dict[str, int]] = init_chainpairdict_zeros(unique_chains) # Count of pairs with Distance < cutoff

    # pDockQ, pDockQ2, LIS scores
    pDockQ_cutoff = 8.0
    pDockQ_scores: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)
    pDockQ_interface_residues: Dict[str, Dict[str, Set[int]]] = init_chainpairdict_set(unique_chains) # Use index for simplicity here

    pDockQ2_scores: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)

    LIS_scores: Dict[str, Dict[str, float]] = init_chainpairdict_zeros(unique_chains)

    # --- Calculate pDockQ ---
    # Note: pDockQ and pDockQ2 use cb_plddt and distances_np which are based
    # on the potentially fewer CB/C3' atoms. The PAE matrix is based on the
    # potentially more numerous CA/C1' tokens. This might need careful indexing
    # if the lists don't align perfectly. Assuming they align for now based on order.
    if num_cb_residues == num_residues: # Check if indexing will work simply
        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 == chain2: continue

                chain1_indices = np.where(chains_np == chain1)[0]
                chain2_indices = np.where(chains_np == chain2)[0]

                interchain_distances = distances_np[np.ix_(chain1_indices, chain2_indices)]
                
                # Find pairs within pDockQ cutoff
                contact_mask = interchain_distances <= pDockQ_cutoff
                num_contact_pairs = np.sum(contact_mask)

                if num_contact_pairs > 0:
                    # Identify unique residues involved in these contacts
                    c1_contact_indices = chain1_indices[np.any(contact_mask, axis=1)]
                    c2_contact_indices = chain2_indices[np.any(contact_mask, axis=0)]
                    
                    contact_residue_indices = np.concatenate((c1_contact_indices, c2_contact_indices))
                    unique_contact_residue_indices = np.unique(contact_residue_indices)
                    
                    # Store unique residue indices for pDockQ pLDDT calculation
                    pDockQ_interface_residues[chain1][chain2] = set(unique_contact_residue_indices)
                    
                    # Calculate mean pLDDT for contacting residues (using cb_plddt)
                    mean_plddt = cb_plddt[list(pDockQ_interface_residues[chain1][chain2])].mean()
                    
                    # pDockQ calculation
                    x_pDockQ = mean_plddt * math.log10(num_contact_pairs)
                    pDockQ_scores[chain1][chain2] = 0.724 / (1 + math.exp(-0.052 * (x_pDockQ - 152.611))) + 0.018
                else:
                    pDockQ_scores[chain1][chain2] = 0.0 # No contacts

        # --- Calculate pDockQ2 ---
        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 == chain2: continue

                chain1_indices = np.where(chains_np == chain1)[0]
                chain2_indices = np.where(chains_np == chain2)[0]

                interchain_distances = distances_np[np.ix_(chain1_indices, chain2_indices)]
                interchain_pae = pae_matrix[np.ix_(chain1_indices, chain2_indices)] # Use PAE for this pair

                # Find pairs within pDockQ cutoff distance
                contact_mask = interchain_distances <= pDockQ_cutoff
                num_contact_pairs = np.sum(contact_mask)

                if num_contact_pairs > 0:
                     # Get PAE values for contacting pairs and apply ptm_func (d0=10 for pDockQ2)
                    contact_pae_values = interchain_pae[contact_mask]
                    ptm_values = ptm_func_vec(contact_pae_values, 10.0)
                    mean_ptm = ptm_values.mean()
                    
                    # Calculate mean pLDDT for contacting residues (should be the same residues as pDockQ)
                    if pDockQ_interface_residues[chain1][chain2]:
                         mean_plddt = cb_plddt[list(pDockQ_interface_residues[chain1][chain2])].mean()
                    else:
                         mean_plddt = 0.0 # Should not happen if num_contact_pairs > 0

                    # pDockQ2 calculation
                    x_pDockQ2 = mean_plddt * mean_ptm
                    pDockQ2_scores[chain1][chain2] = 1.31 / (1 + math.exp(-0.075 * (x_pDockQ2 - 84.733))) + 0.005
                else:
                    pDockQ2_scores[chain1][chain2] = 0.0 # No contacts
    else:
         print("Warning: Number of CA/C1' residues and CB/C3' residues do not match. Skipping pDockQ and pDockQ2 calculations.")
         # Keep pDockQ and pDockQ2 scores as 0.0

    # --- Calculate LIS ---
    # LIS uses PAE <= 12 A
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2: continue

            chain1_indices = np.where(chains_np == chain1)[0]
            chain2_indices = np.where(chains_np == chain2)[0]

            interchain_pae = pae_matrix[np.ix_(chain1_indices, chain2_indices)]

            if interchain_pae.size > 0:
                # Apply the LIS cutoff (PAE <= 12)
                valid_pae_values = interchain_pae[interchain_pae <= 12.0]
                
                if valid_pae_values.size > 0:
                    # Calculate LIS scores (12 - PAE) / 12
                    scores = (12.0 - valid_pae_values) / 12.0
                    LIS_scores[chain1][chain2] = np.mean(scores)
                else:
                    LIS_scores[chain1][chain2] = 0.0 # No pairs with PAE <= 12
            else:
                LIS_scores[chain1][chain2] = 0.0 # No pairs in this chain combination

    # --- Calculate ipSAE variations ---
    # This loop calculates the by-residue ipSAE components and accumulates
    # counts and sums for the chain-pair scores.

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2: continue

            chain1_indices = np.where(chains_np == chain1)[0]
            chain2_indices = np.where(chains_np == chain2)[0]

            # ipTM (d0 based on total residues in chain1 + chain2) - base value
            n0chn[chain1][chain2] = len(chain1_indices) + len(chain2_indices)
            d0chn[chain1][chain2] = calc_d0(n0chn[chain1][chain2], chain_pair_type[chain1][chain2])
            
            # Calculate ptm matrix using d0chn
            ptm_matrix_d0chn = ptm_func_vec(pae_matrix, d0chn[chain1][chain2])

            # Initialize accumulators for asymmetric chain-pair scores
            sum_ipsae_d0chn_asym = 0.0
            sum_ipsae_d0dom_asym = 0.0
            sum_ipsae_d0res_asym = 0.0
            count_valid_residues_asym = 0 # Count chain1 residues that have at least one chain2 residue with PAE < cutoff

            # Calculate by-residue scores (from chain1 perspective to chain2)
            for i in chain1_indices:
                # Indices of chain2 residues
                chain2_indices_for_i = chain2_indices

                # Apply PAE cutoff for ipSAE variants
                valid_pae_mask_for_i = (pae_matrix[i, chain2_indices_for_i] < pae_cutoff)
                chain2_indices_with_low_pae = chain2_indices_for_i[valid_pae_mask_for_i]

                # Calculate n0res (number of chain2 residues interacting with residue i < cutoff)
                n0res_byres[chain1][chain2][i] = np.sum(valid_pae_mask_for_i)

                # Calculate d0res for residue i
                d0res_byres[chain1][chain2][i] = calc_d0(n0res_byres[chain1][chain2][i], chain_pair_type[chain1][chain2])

                # Calculate by-residue scores if there are valid interacting residues
                if chain2_indices_with_low_pae.size > 0:
                    count_valid_residues_asym += 1 # This chain1 residue contributes to the asymmetric sum

                    # ipSAE_d0chn_byres: mean ptm (d0=d0chn) for interacting chain2 residues
                    ipsae_d0chn_byres[chain1][chain2][i] = ptm_matrix_d0chn[i, chain2_indices_with_low_pae].mean()
                    sum_ipsae_d0chn_asym += ipsae_d0chn_byres[chain1][chain2][i]

                    # ipSAE_d0dom_byres: calculate mean ptm using a temporary d0dom for residue i's pair set
                    # d0dom depends on *all* interface residues for the chain pair, calculated after this loop.
                    # This was a potential ambiguity in the original script. Let's use the final d0dom.
                    # For now, calculate ptm using a d0 based on n0res for *this* residue.
                    # The 'true' ipSAE_d0dom_byres will be calculated *after* n0dom/d0dom are finalized.
                    # Let's store a temporary score here and calculate the final one later.
                    # We need the PAE values for the interacting chain2 residues for residue i.
                    pae_values_for_i = pae_matrix[i, chain2_indices_for_i][valid_pae_mask_for_i]
                    # Calculate ptm using d0res for residue i
                    ptm_values_d0res_for_i = ptm_func_vec(pae_values_for_i, d0res_byres[chain1][chain2][i])
                    ipsae_d0res_byres[chain1][chain2][i] = ptm_values_d0res_for_i.mean()
                    sum_ipsae_d0res_asym += ipsae_d0res_byres[chain1][chain2][i]

                    # Track unique residues involved in the interface based on PAE cutoff
                    res_id1 = residue_ids_np[i]
                    pae_interface_residues[chain1][chain2].add(res_id1)
                    
                    res_ids2 = residue_ids_np[chain2_indices_with_low_pae]
                    for res_id2 in res_ids2:
                        pae_interface_residues[chain1][chain2].add(res_id2)
                        
                    pae_valid_pair_counts[chain1][chain2] += chain2_indices_with_low_pae.size

                    # Track unique residues involved in the interface based on Distance cutoff (for PyMOL)
                    # This is separate from the PAE cutoff used for score calculation
                    dist_valid_mask_for_i = (distances_np[i, chain2_indices_for_i] < dist_cutoff)
                    chain2_indices_with_low_dist = chain2_indices_for_i[dist_valid_mask_for_i]
                    
                    if chain2_indices_with_low_dist.size > 0:
                         res_id1 = residue_ids_np[i]
                         dist_interface_residues[chain1][chain2].add(res_id1)
                         
                         res_ids2 = residue_ids_np[chain2_indices_with_low_dist]
                         for res_id2 in res_ids2:
                              dist_interface_residues[chain1][chain2].add(res_id2)

                         dist_valid_pair_counts[chain1][chain2] += chain2_indices_with_low_dist.size

            # Calculate asymmetric chain-pair scores (mean over chain1 residues with >0 interacting chain2 residues)
            if count_valid_residues_asym > 0:
                ipsae_d0chn_asym[chain1][chain2] = sum_ipsae_d0chn_asym / count_valid_residues_asym
                ipsae_d0res_asym[chain1][chain2] = sum_ipsae_d0res_asym / count_valid_residues_asym
            else:
                ipsae_d0chn_asym[chain1][chain2] = 0.0
                ipsae_d0res_asym[chain1][chain2] = 0.0

            # Calculate n0dom (total number of residues in chain1+chain2 with PAE < cutoff to ANY residue in the other chain)
            n0dom[chain1][chain2] = len(pae_interface_residues[chain1][chain2])

            # Calculate d0dom based on n0dom
            d0dom[chain1][chain2] = calc_d0(n0dom[chain1][chain2], chain_pair_type[chain1][chain2])

            # Now calculate the 'true' ipSAE_d0dom_byres and accumulate for the asymmetric sum
            sum_ipsae_d0dom_asym_final = 0.0
            for i in chain1_indices:
                 chain2_indices_for_i = chain2_indices
                 valid_pae_mask_for_i = (pae_matrix[i, chain2_indices_for_i] < pae_cutoff)
                 chain2_indices_with_low_pae = chain2_indices_for_i[valid_pae_mask_for_i]

                 if chain2_indices_with_low_pae.size > 0:
                      # Calculate ptm using the chain-pair-level d0dom
                      pae_values_for_i = pae_matrix[i, chain2_indices_for_i][valid_pae_mask_for_i]
                      ptm_values_d0dom_for_i = ptm_func_vec(pae_values_for_i, d0dom[chain1][chain2])
                      ipsae_d0dom_byres[chain1][chain2][i] = ptm_values_d0dom_for_i.mean()
                      sum_ipsae_d0dom_asym_final += ipsae_d0dom_byres[chain1][chain2][i]
                 else:
                      ipsae_d0dom_byres[chain1][chain2][i] = 0.0 # Ensure it's zero if no interacting residues

            # Final asymmetric ipSAE_d0dom score
            if count_valid_residues_asym > 0:
                 ipsae_d0dom_asym[chain1][chain2] = sum_ipsae_d0dom_asym_final / count_valid_residues_asym
            else:
                 ipsae_d0dom_asym[chain1][chain2] = 0.0


    # --- Calculate Symmetric (Max) ipSAE Scores ---
    # This happens after the asymmetric scores are calculated for all pairs.

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2: continue

            # ipSAE_d0chn_max
            val1 = ipsae_d0chn_asym[chain1].get(chain2, 0.0) # Use .get for safety if a pair wasn't computed
            val2 = ipsae_d0chn_asym[chain2].get(chain1, 0.0)
            ipsae_d0chn_max[chain1][chain2] = max(val1, val2)
            # Determine which chain contributed the max score
            if val1 >= val2:
                 ipsae_d0chn_maxres[chain1][chain2] = f"{chain1} ({val1:.3f})"
            else:
                 ipsae_d0chn_maxres[chain1][chain2] = f"{chain2} ({val2:.3f})"

            # ipSAE_d0dom_max
            val1 = ipsae_d0dom_asym[chain1].get(chain2, 0.0)
            val2 = ipsae_d0dom_asym[chain2].get(chain1, 0.0)
            ipsae_d0dom_max[chain1][chain2] = max(val1, val2)
            if val1 >= val2:
                 ipsae_d0dom_maxres[chain1][chain2] = f"{chain1} ({val1:.3f})"
            else:
                 ipsae_d0dom_maxres[chain1][chain2] = f"{chain2} ({val2:.3f})"

            # ipSAE_d0res_max (referred to as just ipSAE in original script output)
            val1 = ipsae_d0res_asym[chain1].get(chain2, 0.0)
            val2 = ipsae_d0res_asym[chain2].get(chain1, 0.0)
            ipsae_d0res_max[chain1][chain2] = max(val1, val2)
            if val1 >= val2:
                 ipsae_d0res_maxres[chain1][chain2] = f"{chain1} ({val1:.3f})"
            else:
                 ipsae_d0res_maxres[chain1][chain2] = f"{chain2} ({val2:.3f})"

    # --- Prepare Output Data ---

    # Data for the main output file (.txt)
    main_output_data: List[Dict[str, Any]] = []
    for i, chain1 in enumerate(unique_chains):
        for j, chain2 in enumerate(unique_chains):
            if chain1 == chain2: continue
            
            # Ensure pairs are only processed once (e.g., A-B, not B-A again)
            # or decide if you want both A-B and B-A rows. The original script
            # seems to process A-B and then B-A, and the ipSAE columns show max.
            # Let's output A-B and B-A rows for clarity.
            
            row_data = {
                'Chain1': chain1,
                'Chain2': chain2,
                'PairType': chain_pair_type[chain1][chain2],
                'pDockQ': pDockQ_scores[chain1][chain2],
                'pDockQ2': pDockQ2_scores[chain1][chain2],
                'LIS': LIS_scores[chain1][chain2],
                'n0chn': n0chn[chain1][chain2],
                'd0chn': d0chn[chain1][chain2],
                'ipSAE_d0chn_Asym': ipsae_d0chn_asym[chain1][chain2],
                'n0dom': n0dom[chain1][chain2], # n0dom(A,B) is symmetric count of interface residues
                'd0dom': d0dom[chain1][chain2],
                'ipSAE_d0dom_Asym': ipsae_d0dom_asym[chain1][chain2],
                'ipSAE_d0res_Asym': ipsae_d0res_asym[chain1][chain2], # This is the one called ipSAE in original by-res output
                'ipSAE_d0chn_Max': ipsae_d0chn_max[chain1][chain2],
                'ipSAE_d0chn_MaxRes': ipsae_d0chn_maxres[chain1][chain2],
                'ipSAE_d0dom_Max': ipsae_d0dom_max[chain1][chain2],
                'ipSAE_d0dom_MaxRes': ipsae_d0dom_maxres[chain1][chain2],
                'ipSAE_d0res_Max': ipsae_d0res_max[chain1][chain2], # This is the one called ipSAE in original main output
                'ipSAE_d0res_MaxRes': ipsae_d0res_maxres[chain1][chain2],
                'PAE_Interface_Residues_Count': n0dom[chain1][chain2], # Same as n0dom
                'Dist_Interface_Residues_Count': len(dist_interface_residues[chain1][chain2])
            }
            # Include original AF3/Boltz1 chain_pair_iptm if available
            if chain_pair_iptm_data and chain1 in chain_pair_iptm_data and chain2 in chain_pair_iptm_data[chain1]:
                 row_data['Original_ChainPair_ipTM'] = chain_pair_iptm_data[chain1][chain2]
            elif af2 and overall_iptm_af2 != -1.0:
                 # For AF2, the file might contain an overall iptm, but not per-pair.
                 # We could add it here for the first pair or as a separate entry.
                 # Let's add it once overall later in the return data.
                 pass
            
            main_output_data.append(row_data)

    # Data for the by-residue output file (.txt)
    by_residue_output_data: List[Dict[str, Any]] = []
    # This loop iterates through each residue i and reports its scores against chain2
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2: continue

            chain1_indices = np.where(chains_np == chain1)[0]
            chain2_indices = np.where(chains_np == chain2)[0]

            for i in chain1_indices:
                 # Indices of chain2 residues relative to the full token list
                 chain2_indices_for_i = chain2_indices

                 # Apply PAE cutoff for ipSAE variants for this specific residue i
                 valid_pae_mask_for_i = (pae_matrix[i, chain2_indices_for_i] < pae_cutoff)
                 chain2_indices_with_low_pae = chain2_indices_for_i[valid_pae_mask_for_i]

                 n0res_i = n0res_byres[chain1][chain2][i]
                 d0res_i = d0res_byres[chain1][chain2][i]

                 # Calculate mean pLDDT for interacting chain2 residues (for this chain1 residue i)
                 mean_plddt_interacting = 0.0
                 if chain2_indices_with_low_pae.size > 0:
                      # Get original indices from structure for plddt lookup
                      # This assumes the filtered PAE/pLDDT indices map directly
                      # to the residues_data/cb_residues_data indices.
                      # If num_tokens != num_residues, this indexing needs careful review.
                      # Assuming 1:1 mapping of token index to residue index for now.
                      interacting_chain2_residue_indices = chain2_indices_with_low_pae
                      mean_plddt_interacting = plddt[interacting_chain2_residue_indices].mean() # Use plddt based on CA/C1' tokens
                 
                 # Get pLDDT for residue i (chain1)
                 plddt_i = plddt[i] # Use plddt based on CA/C1' tokens

                 row_data = {
                     'AlignChain': chain1, # The chain the residue belongs to
                     'ScoredChain': chain2, # The chain being scored against
                     'AlignResIndex': i, # Internal index in the filtered residue list
                     'AlignResID': residue_ids_np[i], # Formatted residue identifier
                     'AlignRespLDDT': plddt_i,
                     'n0chn': n0chn[chain1][chain2], # Total residues in pair
                     'n0dom': n0dom[chain1][chain2], # Total interface residues in pair (symmetric)
                     'n0res': int(n0res_i), # Interacting residues in chain2 for this chain1 residue
                     'd0chn': d0chn[chain1][chain2],
                     'd0dom': d0dom[chain1][chain2],
                     'd0res': float(d0res_i),
                     'ipSAE_d0chn': ipsae_d0chn_byres[chain1][chain2][i], # Mean ptm (d0=d0chn) for interacting chain2 residues
                     'ipSAE_d0dom': ipsae_d0dom_byres[chain1][chain2][i], # Mean ptm (d0=d0dom) for interacting chain2 residues
                     'ipSAE_d0res': ipsae_d0res_byres[chain1][chain2][i], # Mean ptm (d0=d0res) for interacting chain2 residues (This is "ipSAE" in original by-res output)
                     'Mean_pLDDT_Interacting_Chain2': mean_plddt_interacting # New field: mean pLDDT of interacting chain2 residues
                 }
                 by_residue_output_data.append(row_data)

    # Data for the PyMOL script (.pml)
    # Select residues in the interface based on Distance cutoff (dist_cutoff)
    pymol_interface_residues: Dict[str, Set[Tuple[str, int, Any]]] = {}
    for chain1 in unique_chains:
         for chain2 in unique_chains:
              if chain1 == chain2: continue
              # Add residues from both sides of the interface pair to a single set
              for res_id_tuple in dist_interface_residues[chain1][chain2]:
                   chain_id, res_num, insert = res_id_tuple
                   if chain_id not in pymol_interface_residues:
                        pymol_interface_residues[chain_id] = set()
                   pymol_interface_residues[chain_id].add((res_num, insert)) # Store as (resnum, insert) tuple

    pymol_script_lines: List[str] = [
        "# PyMOL script to visualize interface residues",
        f"# Interface defined by Cbeta/C3' distance < {dist_cutoff} A",
        f"# Interface residues based on PAE < {pae_cutoff} A are listed in the .txt files",
        "hide all",
        "show cartoon",
        "color gray",
        "set cartoon_transparency, 0.5",
        "",
        "# Interface residues:",
    ]
    
    # Sort chains for consistent output
    for chain in sorted(pymol_interface_residues.keys()):
         # Extract only residue numbers (dropping insert code for contiguous_ranges)
         residue_numbers_only = {res_num for res_num, insert in pymol_interface_residues[chain]}
         res_range_string = contiguous_ranges(residue_numbers_only)
         
         if res_range_string:
             pymol_script_lines.append(f"show cartoon, chain {chain} and res {res_range_string}")
             pymol_script_lines.append(f"color blue, chain {chain} and res {res_range_string}")
             pymol_script_lines.append(f"show sticks, chain {chain} and res {res_range_string} and (not element H)")
             pymol_script_lines.append(f"util.cbag chain {chain} and res {res_range_string}")
             pymol_script_lines.append("")

    pymol_script_content = "\n".join(pymol_script_lines)


    # --- Return Results ---

    results: Dict[str, Any] = {
        'output_stem': output_stem,
        'main_scores': main_output_data,
        'by_residue_scores': by_residue_output_data,
        'pymol_script_content': pymol_script_content,
        'interface_residues_pae_cutoff': pae_interface_residues, # Residues with PAE < cutoff
        'interface_residues_dist_cutoff': dist_interface_residues, # Residues with Distance < cutoff
        'unique_chains': unique_chains,
        'pae_cutoff': pae_cutoff,
        'dist_cutoff': dist_cutoff,
        'file_info': {
            'pae_file': pae_file_path,
            'structure_file': structure_file_path,
            'type': 'AF2' if af2 else ('AF3' if af3 else ('Boltz1' if boltz1 else 'Unknown')),
        }
    }
    
    if af2:
         results['file_info']['overall_iptm_af2'] = overall_iptm_af2
         results['file_info']['overall_ptm_af2'] = overall_ptm_af2

    # Include raw chain_pair_iptm if available from AF3/Boltz1
    if chain_pair_iptm_data:
         results['original_chain_pair_iptm'] = chain_pair_iptm_data


    return results

# --- Example Usage (how you would call the function) ---

if __name__ == "__main__":
    # This block demonstrates how to use the function and handle its output
    # It mimics the original script's command-line interface for demonstration.

    print("ipSAE, pDockQ, pDockQ2, LIS Calculator")
    print("https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1")
    print("Roland Dunbrack, Fox Chase Cancer Center")
    print("version 3 (refactored into function)")
    print("April 6, 2025 (original script date)")
    print("-" * 30)

    # Ensure correct usage
    if len(sys.argv) < 5:
        print("Usage for AF2:")
        print("   python ipsae.py <path_to_pae_json_file> <path_to_pdb_file> <pae_cutoff> <dist_cutoff>")
        print("   Example: python script_name.py RAF1_KSR1.json RAF1_KSR1.pdb 10 10")
        print("")
        print("Usage for AF3:")
        print("   python ipsae.py <path_to_pae_json_file> <path_to_mmcif_file> <pae_cutoff> <dist_cutoff>")
        print("   Example: python script_name.py fold_aurka_tpx2_full_data_0.json fold_aurka_tpx2_model_0.cif 10 10")
        print("")
        print("Usage for Boltz1:")
        print("   python ipsae.py <path_to_pae_npz_file> <path_to_mmcif_file> <pae_cutoff> <dist_cutoff>")
        print("   Example: python ipsae.py pae_AURKA_TPX2_model_0.npz AURKA_TPX2_model_0.cif 10 10")
        sys.exit(1)

    pae_input_path = sys.argv[1]
    structure_input_path = sys.argv[2]
    try:
        pae_cutoff_input = float(sys.argv[3])
        dist_cutoff_input = float(sys.argv[4])
    except ValueError:
        print("Error: PAE and distance cutoffs must be numbers.")
        sys.exit(1)

    try:
        # Call the main function
        scores_data = calculate_ipsae_scores(
            pae_file_path=pae_input_path,
            structure_file_path=structure_input_path,
            pae_cutoff=pae_cutoff_input,
            dist_cutoff=dist_cutoff_input
        )

        # --- Write Output Files based on returned data ---
        output_stem = scores_data['output_stem']
        main_file_path = f"{output_stem}.txt"
        by_res_file_path = f"{output_stem}_byres.txt"
        pml_file_path = f"{output_stem}.pml"

        # Write main scores file
        with open(main_file_path, 'w') as OUT:
            # Write header
            header = ["Chain1", "Chain2", "PairType", "pDockQ", "pDockQ2", "LIS",
                      "n0chn", "d0chn", "ipSAE_d0chn_Asym", "n0dom", "d0dom",
                      "ipSAE_d0dom_Asym", "ipSAE_d0res_Asym", # This is the one called ipSAE in original main output
                      "ipSAE_d0chn_Max", "ipSAE_d0chn_MaxRes", "ipSAE_d0dom_Max",
                      "ipSAE_d0dom_MaxRes", "ipSAE_d0res_Max", "ipSAE_d0res_MaxRes",
                      "PAE_Interface_Residues_Count", "Dist_Interface_Residues_Count"]
            if 'Original_ChainPair_ipTM' in scores_data['main_scores'][0] if scores_data['main_scores'] else []:
                 header.append("Original_ChainPair_ipTM")
            elif scores_data['file_info']['type'] == 'AF2' and 'overall_iptm_af2' in scores_data['file_info']:
                 # Add overall AF2 iptm if available (just print it once or add as a note)
                 pass # Decided to add to printed output below instead of file column
                 
            OUT.write("\t".join(header) + "\n")

            # Write data rows
            for row in scores_data['main_scores']:
                row_values = [
                    row['Chain1'], row['Chain2'], row['PairType'],
                    f"{row['pDockQ']:.4f}", f"{row['pDockQ2']:.4f}", f"{row['LIS']:.4f}",
                    str(row['n0chn']), f"{row['d0chn']:.3f}", f"{row['ipSAE_d0chn_Asym']:.4f}",
                    str(row['n0dom']), f"{row['d0dom']:.3f}", f"{row['ipSAE_d0dom_Asym']:.4f}",
                    f"{row['ipSAE_d0res_Asym']:.4f}", # Corresponds to ipSAE in original by-res output column
                    f"{row['ipSAE_d0chn_Max']:.4f}", row['ipSAE_d0chn_MaxRes'],
                    f"{row['ipSAE_d0dom_Max']:.4f}", row['ipSAE_d0dom_MaxRes'],
                    f"{row['ipSAE_d0res_Max']:.4f}", row['ipSAE_d0res_MaxRes'], # Corresponds to ipSAE in original main output
                    str(row['PAE_Interface_Residues_Count']), str(row['Dist_Interface_Residues_Count'])
                ]
                if 'Original_ChainPair_ipTM' in row:
                    row_values.append(f"{row['Original_ChainPair_ipTM']:.4f}")
                OUT.write("\t".join(row_values) + "\n")

        print(f"Main scores written to {main_file_path}")

        # Write by-residue scores file
        with open(by_res_file_path, 'w') as OUT2:
             # Write header - ensure it matches the keys in by_residue_output_data
             header2 = ["AlignChain", "ScoredChain", "AlignResIndex", "AlignResID",
                        "AlignRespLDDT", "n0chn", "n0dom", "n0res", "d0chn", "d0dom",
                        "d0res", "ipSAE_d0chn", "ipSAE_d0dom", "ipSAE_d0res", # ipSAE_d0res is called 'ipSAE' in original output
                        "Mean_pLDDT_Interacting_Chain2"]
             OUT2.write("\t".join(header2) + "\n")

             # Write data rows
             for row in scores_data['by_residue_scores']:
                 row_values = [
                     row['AlignChain'], row['ScoredChain'], str(row['AlignResIndex']),
                     row['AlignResID'], f"{row['AlignRespLDDT']:.2f}", str(row['n0chn']),
                     str(row['n0dom']), str(row['n0res']), f"{row['d0chn']:.3f}",
                     f"{row['d0dom']:.3f}", f"{row['d0res']:.3f}", f"{row['ipSAE_d0chn']:.4f}",
                     f"{row['ipSAE_d0dom']:.4f}", f"{row['ipSAE_d0res']:.4f}", # This column corresponds to ipSAE in the original by-res output
                     f"{row['Mean_pLDDT_Interacting_Chain2']:.2f}"
                 ]
                 OUT2.write("\t".join(row_values) + "\n")

        print(f"By-residue scores written to {by_res_file_path}")

        # Write PyMOL script file
        with open(pml_file_path, 'w') as PML:
            PML.write(scores_data['pymol_script_content'])

        print(f"PyMOL script written to {pml_file_path}")

        # Print summary to console
        print("\n--- Summary ---")
        print(f"Structure File: {scores_data['file_info']['structure_file']}")
        print(f"PAE File: {scores_data['file_info']['pae_file']}")
        print(f"Model Type: {scores_data['file_info']['type']}")
        print(f"PAE Cutoff: {scores_data['pae_cutoff']} A")
        print(f"Distance Cutoff (for PyMOL/count): {scores_data['dist_cutoff']} A")
        print("\nChain Pair Scores:")
        print("Chain1-Chain2 | pDockQ | pDockQ2 | LIS   | ipSAE(d0res,Max) | Original_ipTM")
        print("-" * 80)
        for row in scores_data['main_scores']:
             orig_iptm_str = f"{row.get('Original_ChainPair_ipTM', 'N/A'):.4f}" if isinstance(row.get('Original_ChainPair_ipTM'), (int, float)) else 'N/A'
             print(f"{row['Chain1']}-{row['Chain2']:<8s} | {row['pDockQ']:.4f} | {row['pDockQ2']:.4f} | {row['LIS']:.4f} | {row['ipSAE_d0res_Max']:.4f}           | {orig_iptm_str}")

        if scores_data['file_info']['type'] == 'AF2' and 'overall_iptm_af2' in scores_data['file_info'] and scores_data['file_info']['overall_iptm_af2'] != -1.0:
             print(f"\nOverall AF2 ipTM from JSON/PKL: {scores_data['file_info']['overall_iptm_af2']:.4f}")
             
        print("\nInterface Residues (by Distance Cutoff for PyMOL):")
        for chain in sorted(scores_data['interface_residues_dist_cutoff'].keys()):
             for chain2 in sorted(scores_data['interface_residues_dist_cutoff'][chain].keys()):
                  if chain != chain2:
                       res_set = scores_data['interface_residues_dist_cutoff'][chain][chain2]
                       if res_set:
                            # Collect residues for this pair from *both* chains
                            pair_interface_residues = {res_id for res_id_tuple in scores_data['interface_residues_dist_cutoff'][chain][chain2] for res_id_tuple in scores_data['interface_residues_dist_cutoff'][chain][chain2]} # This isn't quite right, need to get all unique residues involved in *any* contact for the pair
                            
                            # A better way: go through the combined set used for PyMOL, filter by chain
                            chain1_res_ids = sorted([f"{r[0]}{r[1] or ''}" for r in scores_data['pymol_interface_residues'].get(chain, set())])
                            chain2_res_ids = sorted([f"{r[0]}{r[1] or ''}" for r in scores_data['pymol_interface_residues'].get(chain2, set())])

                            if chain1_res_ids or chain2_res_ids:
                                 print(f"  {chain}-{chain2}:")
                                 print(f"    {chain}: {', '.join(chain1_res_ids)}")
                                 print(f"    {chain2}: {', '.join(chain2_res_ids)}")
                       else:
                            print(f"  {chain}-{chain2}: No interface residues found within distance cutoff.")


    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)