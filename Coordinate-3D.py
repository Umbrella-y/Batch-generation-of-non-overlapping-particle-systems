import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import os 
import numpy as np
def generate_tangent_spheres(n, x_min, x_max, y_min, y_max, z_min, z_max):
    spheres = []
    
    def is_tangent(sphere1, sphere2):
        distance = math.sqrt((sphere1[0] - sphere2[0])**2 + (sphere1[1] - sphere2[1])**2 + (sphere1[2] - sphere2[2])**2)
        return distance <= sphere1[3] + sphere2[3]
    
    def is_within_bounds(sphere):
        return (
            sphere[0] - sphere[3] >= x_min and sphere[0] + sphere[3] <= x_max and
            sphere[1] - sphere[3] >= y_min and sphere[1] + sphere[3] <= y_max and
            sphere[2] - sphere[3] >= z_min and sphere[2] + sphere[3] <= z_max
        )
    
    def generate_sphere():
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(z_min, z_max)
        radius = 25#random.uniform(min(x_max - x_min, y_max - y_min, z_max - z_min) / 10, min(x_max - x_min, y_max - y_min, z_max - z_min) / 2)
        return (x, y, z, radius)
    
    def spheres_overlap(sphere):
        for existing_sphere in spheres:
            if is_tangent(sphere, existing_sphere) or not is_within_bounds(sphere):
                return True
        return False
    
    iteration = 0
    while len(spheres) < n and iteration <= 100000:
        iteration += 1
        print("尝试创建球：{}".format(len(spheres)), "已经迭代次数：{}".format(iteration), end='\r')
        sphere = generate_sphere()
        if not spheres_overlap(sphere):
            spheres.append(sphere)
    
    return spheres
def plot_spheres(spheres, x_min, x_max, y_min, y_max, z_min, z_max):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for sphere in spheres:
        ax.scatter(sphere[0], sphere[1], sphere[2], s=sphere[3]**2, edgecolors='r', facecolors='none')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    plt.savefig('profile.png')
def read_atom_data(data_file):
    with open(data_file, 'r') as file:
        lines = file.readlines()

    atoms_line = lines[2].split()
    num_atoms = int(atoms_line[0])

    x_range = lines[5].split()
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_range = lines[6].split()
    y_min, y_max = float(y_range[0]), float(y_range[1])
    z_range = lines[7].split()
    z_min, z_max = float(z_range[0]), float(z_range[1])

    masses_line = lines[11].split()
    atom_type = int(masses_line[0])
    atom_mass = float(masses_line[1])

    atom_lines = lines[15:15+num_atoms]
    columns = ['atom_id', 'atom_type', 'x', 'y', 'z']
    if 'image0' in lines[15] and 'image1' in lines[15] and 'image2' in lines[15]:
        columns.extend(['image0', 'image1', 'image2'])

    data = []
    for line in atom_lines:
        atom_info = line.split()
        atom_id = int(atom_info[0])
        atom_type = int(atom_info[1])
        x = float(atom_info[2])
        y = float(atom_info[3])
        z = float(atom_info[4])
        if len(atom_info) > 5:
            image0 = int(atom_info[5])
            image1 = int(atom_info[6])
            image2 = int(atom_info[7])
            data.append([atom_id, atom_type, x, y, z, image0, image1, image2])
        else:
            data.append([atom_id, atom_type, x, y, z])

    df = pd.DataFrame(data, columns=columns)
    return df, x_min, x_max, y_min, y_max, z_min, z_max, atom_mass

def check_atoms_in_spheres(df, spheres):
    df['in_sphere'] = False
    for i, row in df.iterrows():
        print("目前正在判定原子：{}".format(i), end='\r')
        atom_x = row['x']
        atom_y = row['y']
        atom_z = row['z']
        for sphere in spheres:
            sphere_x = sphere[0]
            sphere_y = sphere[1]
            sphere_z = sphere[2]
            sphere_radius = sphere[3]
            distance = (atom_x - sphere_x)**2 + (atom_y - sphere_y)**2 + (atom_z - sphere_z)**2
            if distance <= sphere_radius**2:
                df.at[i, 'in_sphere'] = True
                break

    return df[df['in_sphere']]

def format_data_file(df, x_min, x_max, y_min, y_max, z_min, z_max, output_file):
    with open(output_file, 'w') as file:
        file.write("Generated data\n\n")
        file.write("\n")
        file.write(" {} atoms\n".format(len(df)))
        file.write(" 1 atom types\n")
        file.write("\n")
        file.write(" {:.3f} {:.3f} xlo xhi\n".format(x_min, x_max))
        file.write(" {:.3f} {:.3f} ylo yhi\n".format(y_min, y_max))
        file.write(" {:.3f} {:.3f} zlo zhi\n".format(z_min, z_max))
        file.write("\n")
        file.write("Masses\n")
        file.write("\n")
        file.write(" 1 {:.2f}\n".format(atom_mass))
        file.write("\n")
        file.write("Atoms # atomic\n")
        file.write("\n")
        for i, row in df.iterrows():
            atom_id = row['atom_id']
            atom_type = row['atom_type']
            x = row['x']
            y = row['y']
            z = row['z']
            if 'image0' in row and 'image1' in row and 'image2' in row:
                image0 = row['image0']
                image1 = row['image1']
                image2 = row['image2']
                file.write(" {} {} {:.6f} {:.6f} {:.6f} {} {} {}\n".format(atom_id, atom_type, x, y, z, image0, image1, image2))
            else:
                file.write(" {} {} {:.6f} {:.6f} {:.6f}\n".format(atom_id, atom_type, x, y, z))

data_file_list = os.listdir(r'/Volumes/新加卷/xxxxx/颗粒种子/8-particle-sinter/output_data')
main_dir = '/Volumes/新加卷/xxxxxx/颗粒种子/8-particle-sinter/'
width = 100
x_min = -width
x_max = width
y_min = -width
y_max = width
z_min = -width
z_max = width

number = 12
sample_num = 10
sintering_temperature = [300,400,500,600,700,800,900,1000] 
for sintering_temperature in sintering_temperature:
    temp_dir = str(sintering_temperature)
    os.mkdir(temp_dir)
    os.chdir(temp_dir)
    name = list()
    for i in range(1,sample_num+1):
        name.append(i)
    print(name)
    datafile = main_dir+'/'+'output_data'
    data_dir = main_dir+'/'+temp_dir
    os.system('cp -r {} {}'.format(datafile,data_dir))
    for i in range(1,sample_num+1):
        spheres = generate_tangent_spheres(number, x_min, x_max, y_min, y_max, z_min, z_max)
        #print("球的坐标和半径:")
        #plot_spheres(spheres, x_min, x_max, y_min, y_max, z_min, z_max)
        with open ('Coordinate_list.txt', 'w') as file:
            for item in spheres:
                file.write(f'{item}\n')
        dirname = str(name[i-1])
        infilename = 'in.8sinter_'+str(name[i-1])+'.lmp'
        os.mkdir(dirname)
        os.chdir(dirname)
        potenfile = main_dir+'/'+'Cu_mishin1.eam.alloy'
        poten_dir = main_dir+'/'+temp_dir+'/'+dirname
        os.system('cp {} {}'.format(potenfile,poten_dir))
        os.system('cp {} {}'.format(potenfile,data_dir))
        with open(infilename, 'w') as file:
            file.write("# 读取库中包含的data文件\n")
            file.write("shell cd ..\n")
            file.write("shell cd output_data\n")
            file.write("#------------------------------模型基本设置\n")
            file.write("clear\nunits metal\ndimension 3\natom_style atomic\natom_modify map array\nboundary p p p\n")
            file.write("#------------------------------变量设置\n")
            for index in range(1,len(spheres)+1):
                file.write("variable r{} equal floor(random(1,20,8123))\n".format(int(index)))
            file.write("#------------------------------模型建立\n")
            file.write("lattice fcc 3.615\n")
            file.write(f"region box block -{width} {width} -{width} {width} -{width} {width} units box\n")
            file.write("create_box 2 box\n")
            for index, data in enumerate(spheres, start=1):
                file.write(f"read_data random_data_${{r{index}}}.data add append shift {data[0]} {data[1]} {data[2]}\n")
            file.write("#------------------------------开始模拟\n")
            file.write("pair_style eam/alloy\n")
            file.write("pair_coeff * * Cu_mishin1.eam.alloy Cu Cu\n")
            file.write("variable seed1 equal floor(random(0,19991227,12393))\n")
            file.write("velocity all create 300 ${seed1} dist gaussian\n")
            file.write("neighbor 2.0 bin\n")
            file.write("neigh_modify every 5 delay 0 check yes\n")
            file.write("thermo 100\n")
            file.write("thermo_style custom step temp press vol ke pe\n")
            file.write("timestep 0.001\n")
            file.write("shell cd ..\n")
            file.write(f"write_data used_sample{str(name[i-1])}.data\n")#需要根据每个循环修改in文件的内容
            file.write(f"shell cd {str(name[i-1])}\n")
            #file.write("shell mkdir dumpfile\n")
            file.write("fix allmo all momentum 1 linear 1 1 1 angular\n")
            file.write(f"fix mynpt all nvt temp 300 {sintering_temperature} 0.1# iso 0 0 1\n")
            #file.write(f"shell mkdir {sintering_temperature}\n")
            if sintering_temperature == 300:
                run_time = 10000
            else:
                run_time = (sintering_temperature-300)*1000
            file.write(f"run {run_time}\n")
            file.write("unfix mynpt\n")
            file.write(f"fix mynpt all nvt temp {sintering_temperature} {sintering_temperature} 0.1# iso 0 0 1\n")
            file.write(f"dump a all custom 10000 *.atom id type x y z\n")
            file.write("run 3500000\n")
            file.write("#------------------------------冷却得到模型\n")
            file.write("unfix mynpt\n")
            file.write(f"fix mynpt all nvt temp {sintering_temperature} 300 0.1# iso 0 0 1\n")
            file.write(f"run {run_time}\n")
            #file.write("shell mkdir sintered_datas\n")
            file.write("shell cd ..\n")
            file.write(f"write_data sintered_sample{str(name[i-1])}.data\n")
            if int(name[i-1]) == 10:
                pass
            else:
                file.write(f"shell cd {str(name[i])}\n")
                file.write(f"jump in.8sinter_{str(name[i])}.lmp\n")
        os.chdir('..')
    os.chdir(main_dir)

        # # 调用函数读取data文件并存储到DataFrame中
        # data_file = r'/Volumes/新加卷/codes/Bubble-Bath/relaxed.data'  # 替换为您的data文件路径
        # df, x_min, x_max, y_min, y_max, z_min, z_max, atom_mass = read_atom_data(data_file)

        # # 指定xyz空间的大小和生成的球的数量
        # n = 15

        # spheres = generate_tangent_spheres(n, x_min, x_max, y_min, y_max, z_min, z_max)
        # print("球的坐标和半径:")
        # plot_spheres(spheres, x_min, x_max, y_min, y_max, z_min, z_max)

        # # 在之前创建的球中检查原子是否存在
        # df_in_sphere = check_atoms_in_spheres(df, spheres)

        # # 输出重新格式化后的data文件
        # output_file = r'/Volumes/新加卷/codes/Bubble-Bath/output_data.data'  # 替换为输出的data文件路径
        # format_data_file(df_in_sphere, x_min, x_max, y_min, y_max, z_min, z_max, output_file)
