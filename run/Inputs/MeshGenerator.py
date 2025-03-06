import meshio

mesh = meshio.read('SphereGmsh.msh')

mesh.write('SphereMesh.vtk')

mesh = meshio.read('SphereGmsh.msh')
print(mesh.cells_dict['tetra'])

