import os

folder = './app_mults/evo_selected/mult7u'
# folder = './app_mults/evo_selected/mult8u'
# folder = './app_mults/vert_trunc/mult7u/'
# folder = './app_mults/vert_trunc/mult8u/'
# folder = './app_mults/vert_trunc/mult6u/'
# folder = './app_mults/resub_als/'
files = os.listdir(folder)
for file in sorted(files):
    if file.endswith('_sop.blif'):
        comm = f'./simulator.out --appMult {folder + '/' + file} > {folder + '/' + file.replace("_sop.blif", "_lutfp.txt")}'
        print(comm)
        # os.system(comm)

hwss = [2, 4, 8, 16, 32]
for hws in hwss:
    for file in sorted(files):
        if file.endswith('_sop.blif'):
            comm = f'python script/gen_bp_lut.py -f {folder + '/' + file.replace("_sop.blif", "_lutfp.txt")} -w {hws} -x {hws} > {folder + '/' + file.replace("_sop.blif", f"_lutfp+bp_avg_{hws}_{hws}.txt")}'
            print(comm)
            # os.system(comm)