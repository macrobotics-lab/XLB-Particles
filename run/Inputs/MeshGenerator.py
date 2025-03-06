import meshio

mesh = meshio.read('SphereGmsh.msh')

print(mesh.points)
print(mesh.cells_dict)