import QC_analysis as QC
import numpy as np
import matplotlib.pyplot as plt


def adjust(coordinates, normal_mode, scaler):
    adjusted_coor = coordinates + normal_mode.dot(np.array(scaler))
    return adjusted_coor.reshape(dat.num_atoms, 3)


def write_xyz(geo, numAtoms, elements, filename):
    geoArray = geo.reshape(numAtoms, 3)
    name = ''
    with open(filename + ".xyz", 'w') as xyz_file:
        xyz_file.write("%d\n%s\n" % (numAtoms, name))
        for atom_ind in range(numAtoms):
            xyz_file.write("{:4} {:11.6f} {:11.6f} {:11.6f}\n".format(
                elements[atom_ind], geoArray[atom_ind, 0], geoArray[atom_ind, 1], geoArray[atom_ind, 2]))


def build_mol_figure(geo):
    ax.cla()
    radiiScale = 1000  # sets the size of the nuclei in the plot
    distances_from1 = geo[:, np.newaxis, :] - geo
    all_distances = np.sqrt(np.einsum("ijk,ijk->ij", distances_from1, distances_from1))

    radii_list = [atomic_radii[element] for element in dat.elements]
    radii_list = np.array(radii_list)
    distance_bond = (radii_list[:, np.newaxis] + radii_list) * 1.5
    adj_matrix = np.logical_and(0.15 < all_distances, distance_bond > all_distances).astype(int)

    adj_list = {}
    for i, j in zip(*np.nonzero(adj_matrix)):
        adj_list.setdefault(i, set()).add(j)
        adj_list.setdefault(j, set()).add(i)

    adjascent_atoms = (
        (atom, neighbour)
        for atom, neighbours in adj_list.items()
        for neighbour in neighbours
    )

    for i, j in adjascent_atoms:
        x = np.append(geo[i, 0], geo[j, 0])
        y = np.append(geo[i, 1], geo[j, 1])
        z = np.append(geo[i, 2], geo[j, 2])
        ax.plot(x, y, z, label='parametric curve', color='k')

    elements = np.unique(dat.elements)
    for element in elements:
        # find indexes of current element
        el_indx = np.where(dat.elements == element)[0]
        # create a geometry of all nuclei of the current element
        el_geo = geo[el_indx, :]
        # Find color for current element. If element does not have a defined color, it sets the color to red.
        # Define colors in 'element_colors' dictionary.
        el_color = element_colors.get(element, 'red')
        # plot nuclei
        ax.scatter(el_geo[:, 0], el_geo[:, 1], el_geo[:, 2], color=el_color, s=atomic_radii[element] * radiiScale)

    limit=1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.xaxis.pane.set_edgecolor('w'), ax.yaxis.pane.set_edgecolor('w'),
    # Remove the axes
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_box_aspect([1,1,1])


kToAU = 4.5563323e-06

atomic_radii = dict(
    Ac=1.88,
    Ag=1.59,
    Al=1.35,
    Am=1.51,
    As=1.21,
    Au=1.50,
    B=0.83,
    Ba=1.34,
    Be=0.35,
    Bi=1.54,
    Br=1.21,
    C=0.68,
    Ca=0.99,
    Cd=1.69,
    Ce=1.83,
    Cl=0.99,
    Co=1.33,
    Cr=1.35,
    Cs=1.67,
    Cu=1.52,
    D=0.23,
    Dy=1.75,
    Er=1.73,
    Eu=1.99,
    F=0.64,
    Fe=1.34,
    Ga=1.22,
    Gd=1.79,
    Ge=1.17,
    H=0.23,
    Hf=1.57,
    Hg=1.70,
    Ho=1.74,
    I=1.40,
    In=1.63,
    Ir=1.32,
    K=1.33,
    La=1.87,
    Li=0.68,
    Lu=1.72,
    Mg=1.10,
    Mn=1.35,
    Mo=1.47,
    N=0.68,
    Na=0.97,
    Nb=1.48,
    Nd=1.81,
    Ni=1.50,
    Np=1.55,
    O=0.68,
    Os=1.37,
    P=1.05,
    Pa=1.61,
    Pb=1.54,
    Pd=1.50,
    Pm=1.80,
    Po=1.68,
    Pr=1.82,
    Pt=1.50,
    Pu=1.53,
    Ra=1.90,
    Rb=1.47,
    Re=1.35,
    Rh=1.45,
    Ru=1.40,
    S=1.02,
    Sb=1.46,
    Sc=1.44,
    Se=1.22,
    Si=1.20,
    Sm=1.80,
    Sn=1.46,
    Sr=1.12,
    Ta=1.43,
    Tb=1.76,
    Tc=1.35,
    Te=1.47,
    Th=1.79,
    Ti=1.47,
    Tl=1.55,
    Tm=1.72,
    U=1.58,
    V=1.33,
    W=1.37,
    Y=1.78,
    Yb=1.94,
    Zn=1.45,
    Zr=1.56,
    Xe=2
)

atomic_mass = dict(
    H=1.00784,
    C=12.011,
    O=15.999,
    F=18.998403,
    Xe=131.293
)

# add colors as needed
element_colors = dict(
    H='grey',
    C='black',
    O='blue',
    F='green',
    Xe='orange'
)

dat = QC.ImportData(input("Input file name: "))
dat.import_freq_data()

mode = int(input("Mode to adjust: "))  # The mode which you want to displace along.
displ = float(input("Displacement amount: ")) # amount that you want to displace.

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.view_init(90, 90, 0)  # top-down view
geo_adjusted = adjust(dat.geometry, dat.mw_normal_modes[0][:, mode + 6], displ)
build_mol_figure(geo_adjusted)
plt.show()
output_name = input(".xyz file name (don't add .xyz to name;\n leave blank to not save a .xyz file): ")
if output_name != "":
    write_xyz(geo_adjusted, dat.num_atoms, dat.elements, output_name)
    
