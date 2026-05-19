project('pyoculus', 'c', version : '1.0.0',  meson_version: '>= 0.64.0',
)

add_languages('fortran', native: false)

fortran_compiler = meson.get_compiler('fortran')
message('fortran compiler id is ' + fortran_compiler.get_id())


f90flags = []
if fortran_compiler.get_id() == 'gcc'
    f90flags += '-ffree-line-length-none'
    message('adding -ffree-line-length-flag')
endif

f90flags += '-O3'
add_global_arguments(f90flags, language: 'fortran')
    

py3 = import('python').find_installation(pure: false)
py_dep = py3.dependency()

incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

sourcefiles = files(
    'pyoculus/maps/SPECfortran/pyvariables.f90',
    'pyoculus/maps/SPECfortran/pybasefn.f90',
    'pyoculus/maps/SPECfortran/pycoords.f90',
    'pyoculus/maps/SPECfortran/pybfield.f90',
    'pyoculus/maps/SPECfortran/pyPJH.f90'
)

module_name = 'pyoculus_spec_fortran_module'


pyoculus_spec_fortran_module_source = custom_target('pyoculus_spec_fortran_module.c',
  input : sourcefiles,  
  output : [module_name+'module.c', module_name+'-f2pywrappers2.f90'],
  command : [py3, '-m', 'numpy.f2py', '@INPUT@', '-m', module_name, '--lower']
)

py3.extension_module(module_name, 
  [sourcefiles, pyoculus_spec_fortran_module_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)

py3.install_sources(
   'pyoculus/__init__.py',
    subdir: 'pyoculus' 
    )

py3.install_sources(
  'pyoculus/fields/__init__.py',
  'pyoculus/fields/cylindrical_bfield.py',
  'pyoculus/fields/cylindrical_bfield_analytic.py',
  'pyoculus/fields/m3dc1_bfield.py',
  'pyoculus/fields/magnetic_field.py',
  'pyoculus/fields/qfm_bfield.py',
  'pyoculus/fields/simsopt_bfield.py',
  'pyoculus/fields/spec_bfield.py',
  'pyoculus/fields/toroidal_bfield.py',
  'pyoculus/fields/two_waves.py',
  'pyoculus/fields/cylindrical_grid_interpolated_field.py',
  'pyoculus/fields/spectre_bfield.py',
  'pyoculus/fields/__init__.py',
  subdir: 'pyoculus/fields'
)

py3.install_sources(
  'pyoculus/geo/interpolate_coordinates.py',
  'pyoculus/geo/__init__.py',
  subdir: 'pyoculus/geo'
)

py3.install_sources(
  'pyoculus/integrators/base_integrator.py',
  'pyoculus/integrators/scipyode_integrator.py',
  'pyoculus/integrators/sdc_integrator.py',
  'pyoculus/integrators/__init__.py',
  subdir: 'pyoculus/integrators'
)

py3.install_sources(
  'pyoculus/maps/base_map.py',
  'pyoculus/maps/cylindrical_bfield_section.py',
  'pyoculus/maps/integrated_map.py',
  'pyoculus/maps/spec_pjh.py',
  'pyoculus/maps/tokamap.py',
  'pyoculus/maps/standard_map.py',
  'pyoculus/maps/toroidal_bfield_section.py',
  'pyoculus/maps/__init__.py',
  subdir: 'pyoculus/maps'
)

py3.install_sources(
  'pyoculus/solvers/base_solver.py',
  'pyoculus/solvers/fixed_point.py',
  'pyoculus/solvers/flux_surface_gr.py',
  'pyoculus/solvers/lyapunov_exponent.py',
  'pyoculus/solvers/manifold.py',
  'pyoculus/solvers/poincare_plot.py',
  'pyoculus/solvers/qfm.py',
  'pyoculus/solvers/resonance_zone.py',
  'pyoculus/solvers/__init__.py',
  subdir: 'pyoculus/solvers'
)

py3.install_sources(
  'pyoculus/toybox/cylindrical_toybox.py',
  'pyoculus/toybox/__init__.py',
  subdir: 'pyoculus/toybox'
)

py3.install_sources(
  'pyoculus/utils/continued_fraction.py',
  'pyoculus/utils/plot.py',
  'pyoculus/utils/cyl_cart_transform.py',
  'pyoculus/utils/convergence_domain.py',
  'pyoculus/utils/fortran.py',
  'pyoculus/utils/__init__.py',
  subdir: 'pyoculus/utils'
)

# Verify the build
#run_command(py3, '-c', 'import ' + module_name, check: true)
