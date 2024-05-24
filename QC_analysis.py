import re
import numpy as np
from numpy import linalg as LA

### Unit conversions ###

bohr2m = 0.529177249e-10 
hartree2joule = 4.35974434e-18
speed_of_light = 299792458
avogadro = 6.0221413e+23
auforce2newton = 8.238726e-08
vib_constant = np.sqrt((avogadro * hartree2joule * 1000) / (bohr2m**2)) / (2 * np.pi * speed_of_light * 100)
k2AU = 4.5563323e-06
hbar = 1.05457182e-34
kToHz = 2 * np.pi * speed_of_light * 100
amu2AU = 1.6601e-27 / 9.1094e-31

###

### "shaper" functions re-shape the Qchem output into a matrix format ###

# re-shapes the force data
def force_shaper(elements, shape):
    n_columns = shape[0]
    elements_per_column = shape[1]
    columns_per_block = shape[2]
    elements_per_block = int(columns_per_block * elements_per_column)
    total_elements = int(n_columns * elements_per_column)
    ordered_blocks = int(total_elements // elements_per_block)
    unordered_columns = int((total_elements - ordered_blocks * elements_per_block) / elements_per_column)
    stack = elements[0:elements_per_block].reshape((elements_per_column, columns_per_block)).flatten(order='F')
    for j in range(ordered_blocks - 1):
        j += 1
        block = elements[elements_per_block * j:elements_per_block * (j + 1)].reshape(
            (elements_per_column, columns_per_block)).flatten(order='F')
        stack = np.hstack((stack, block))
    if unordered_columns > 0:
        block = elements[elements_per_block * (j + 1):].reshape((elements_per_column, unordered_columns)).flatten(order='F')
        stack = np.hstack((stack, block))
    return stack


# re-shapes the Hessian matrix output into a NxN matrix
def hessian_shaper(elements, shape):
    n_columns = shape[0]
    elements_per_column = shape[1]
    columns_per_block = shape[2]
    elements_per_block = int(columns_per_block * elements_per_column)
    total_elements = int(n_columns * elements_per_column)
    ordered_blocks = int(total_elements // elements_per_block)
    unordered_columns = int((total_elements - ordered_blocks * elements_per_block) / elements_per_column)
    stack = elements[0:elements_per_block].reshape((elements_per_column, columns_per_block))
    for j in range(ordered_blocks - 1):
        j += 1
        block = elements[elements_per_block * j:elements_per_block * (j + 1)].reshape(
            (elements_per_column, columns_per_block))
        stack = np.hstack((stack, block))
    if unordered_columns > 0:
        block = elements[elements_per_block * (j + 1):].reshape((elements_per_column, unordered_columns))
        stack = np.hstack((stack, block))
    return stack


# Shapes the nuclear geometry output into 1D array
def geo_shaper(elements, shape):
    return elements.flatten(order='F')


def ir_shaper(elements, shape):
    return elements  # no manipulation necessary

###

### Molecular data object ###

# Parses through .out file to find text given by regex expression "key_word"
def data_collector(file_name, key_word, shaper, shape):
    file = open(file_name, "r")
    txt = file.read()
    selections = re.findall(key_word, txt)
    file.close()
    out_array = []
    for i, selection in enumerate(selections):
        selection = np.array(re.findall('\s(-?\d*\.\d*)\s', selection)).astype(
            float)  # create an array of all floats in selection in order from left to right and top to bottom (like reading a book).
        shaped_select = shaper(selection, shape)
        out_array.append(shaped_select)
    return out_array


# Finds the mass of each nuclei
def mass_finder(file_name, n_atoms):
    file = open(file_name, "r")
    txt = file.read()
    masses = np.array(re.findall('Has Mass\s+(\d*\.\d*)\s+', txt)).astype(float)
    file.close()
    return masses.reshape((int(masses.size/n_atoms), n_atoms))


# Finds the type of element each nuclei is
def element_finder(file_name, key_word):                                                                    
    file = open(file_name, "r")                                                                             
    txt = file.read()                                                                                       
    selection = re.findall(key_word, txt)[0]                                                                
    file.close()                                                                                            
    return np.array(re.findall('\s([A-Za-z]{1,2})\s', selection)) 


# Finds the reduced mass of each vibrational mode, stored in order of increasing frequency
def reduced_mass(mass, L):
    m_i = np.array([mass[i // 3] for i in range(mass.size * 3)])
    return 1 / np.sum(L.T ** 2 / m_i, 1)


# Creates object containing force and/or frequency data of a specific molecule. .out files may be loaded on initiation
# by setting the "file_name" argument to a list of .out files. After initiation, further .out files may be loaded by
# calling "import_freq_data" or "import_force_data".
class ImportData:
    def __init__(self, file_name, geo_file_num=0, geo_num=-1):
        # contains .out files loaded
        self.my_file = []
        # checks if file is already loaded and if not appends it to "my_file"
        if isinstance(file_name, list):
            self.my_file.extend(file_name)
        else:
            self.my_file.append(file_name)
        # vector of cartesian nuclear coordinates
        self.geometry = data_collector(self.my_file[geo_file_num], geo_key, geo_shaper, [])[geo_num]
        # number of nuclei
        self.num_atoms = int(self.geometry.size / 3)
        # mass of each nuclei in same order as given in the .out file
        self.masses = mass_finder(self.my_file[geo_file_num], self.num_atoms)
        # list of elements in the same order as the masses
        self.elements = element_finder(self.my_file[geo_file_num], geo_key)  
        # Details the shape of the Hessian matrix data in the .out file. Used for shaper function.
        self.hessian_shape = [3 * self.num_atoms,  # number of normal modes
                              3 * self.num_atoms,  # number of elements per normal mode
                              6]  # number of force vectors per block as organized by QChem output
        # Mass weighted frequencies from smallest to largest. Includes translational and rotational modes but the
        # absolute value is taken so no complex numbers are given. Check .out file for imaginary frequencies.
        self.mw_freq = []
        # Array of normal modes projected on the cartesian nuclear coordinates. Ordering corresponds to ordering of
        # mass weigthed frequencies.
        self.mw_normal_modes = []
        # reduced masses with same ordering as mass weighted frequencies
        self.red_mass = []
        # infrared intensities with same ordering as mass weighted frequencies
        self.ir_int = []
        # Details the shape of the force data in the .out file. Used for shaper function.
        self.force_shape = [self.num_atoms,  # number of atoms
                            3,  # number of elements of the force vector for each atom (force along x,y,z)
                            6]  # number of force vectors per block as organized by QChem output
        # contains the forces acting on the nuclei in cartesian nuclear coordinates with same ordering as mass
        # weighted frequencies
        self.force = []
        # Huang-Rhys factors calculated from the forces using "calc_hr" function.
        self.hr = []
    # Imports relevant from QChem frequency calculation
    def import_freq_data(self, file_number=0):
        self.mw_hessian = data_collector(self.my_file[file_number], hessian_key, hessian_shaper, self.hessian_shape)
        for n, hess in enumerate(self.mw_hessian):
            v, u = LA.eigh(hess)
            self.mw_normal_modes.append(u)
            self.mw_freq.append(np.abs(np.sqrt(v.astype(complex))) * vib_constant)
            self.red_mass.append(reduced_mass(self.masses[n], u))
        ir_int = data_collector(self.my_file[file_number], ir_key, ir_shaper, [])
        ir_int = np.hstack(ir_int)
        ir_int = ir_int.reshape((int(ir_int.size/(3 * self.num_atoms - 6)), int(3 * self.num_atoms - 6)))
        self.ir_int = np.zeros(ir_int.shape + np.array([0, 6]))
        self.ir_int[:, 6:] = ir_int
    # Calculates relevant data from QChem force calculation
    def import_force_data(self, file_number=1):
        self.force = data_collector(self.my_file[file_number], force_key, force_shaper, self.force_shape)
    # Calculates the Huang-Rhys factors associated with a certain transition for each normal mode. Requires a
    # frequency calculation from the initial state (or whichever state you want the coordinates to be defined in)
    # and a force calculation in the final state.
    def calc_hr(self, force_indx=0, freq_indx=0):
        force = self.force[force_indx]
        Q = self.mw_normal_modes[freq_indx]
        freq = self.mw_freq[freq_indx]
        mass = self.masses[freq_indx]
        mass_matrix = lambda m: np.diag(np.array([1 / (m[i // 3]) ** 0.5 for i in range(m.size * 3)]))
        M_kg = mass_matrix(mass * amu2AU)
        force_J_per_m = force
        mass_force = M_kg.dot(force_J_per_m)
        force_norm = Q.T.dot(mass_force)
        self.hr.append(force_norm ** 2 / (2 * (freq * k2AU) ** 3))
    # Adjusts the coordinates along the specified normal mode by a scaler 'scale'
    def adjust(coordinates, normal_mode, scaler):
        adjusted_coor = coordinates + normal_mode.dot(np.array(scaler))
        return adjusted_coor.reshape(nAtoms, 3)

###
        
### key words for capturing strings from the .out file ###

# captures force data
force_key = '\s+ Gradient of the state energy \(including CIS Excitation Energy\)([\s\S]*?)[^-{1}\s\d\.\w{1}]'
# captures mass-weighted Hessian matrix
hessian_key = '\s+ Mass-Weighted Hessian Matrix:([\s\S]*?)[^-\s\d\.]'
# captures geometry
geo_key = '\s+ Standard Nuclear Orientation \(Angstroms\)[\s\S]*?-+([\s\S]*?)[^-{1}\s\d\.\w{1}]'
# captures the infrared intensities of each normal mode
ir_key = 'IR Intens:([\s\S]*?)[^\s\d\.]'


