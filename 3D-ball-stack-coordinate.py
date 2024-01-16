import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import os 
import numpy as np

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

def generate_tangent_spheres(n, x_min, x_max, y_min, y_max, z_min, z_max):
    
    def is_tangent(sphere1, sphere2):
        distance = math.sqrt((sphere1[0] - sphere2[0])**2 + (sphere1[1] - sphere2[1])**2 + (sphere1[2] - sphere2[2])**2)
        return distance <= sphere1[3] + sphere2[3]
    
    def is_within_bounds(sphere):
        return (
            sphere[0] - sphere[3] >= x_min and sphere[0] + sphere[3] <= x_max and
            sphere[1] - sphere[3] >= y_min and sphere[1] + sphere[3] <= y_max and
            sphere[2] - sphere[3] >= z_min and sphere[2] + sphere[3] <= z_max
        )
    
    def generate_sphere(radius):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(z_min, z_max)
        radius = radius#random.uniform(min(x_max - x_min, y_max - y_min, z_max - z_min) / 10, min(x_max - x_min, y_max - y_min, z_max - z_min) / 2)
        return (x, y, z, radius)
    def generate_connected_sphere(spheres,radius):
        reference_sphere = random.choice(spheres)
        direction = normalize_vector(np.random.rand(3))
        x = reference_sphere[0] + direction[0] * (2 * radius)
        y = reference_sphere[1] + direction[1] * (2 * radius)
        z = reference_sphere[2] + direction[2] * (2 * radius)
        radius = radius 
        return (x, y, z, radius)
    def normalize_vector(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    def spheres_overlap(sphere):
        for existing_sphere in spheres:
            if is_tangent(sphere, existing_sphere) or not is_within_bounds(sphere):
                return True
        return False
    spheres = []
    while len(spheres) <n:
        spheres = []
        iteration = 0
        # 生成第一个球的信息
        radius = 25

        while len(spheres) == 0:
            sphere = generate_sphere(radius)
            if is_within_bounds(sphere):
                spheres.append(sphere)
        # 生成剩余的球的信息
        while len(spheres) <n and iteration <= 10000:
            iteration += 1
            sphere = generate_connected_sphere(spheres, radius)
            if iteration % 10000 ==0:
                print("尝试创建连接球：{}".format(len(spheres)), "已经迭代次数：{}".format(iteration), end='\r')
            if not spheres_overlap(sphere):
                spheres.append(sphere)
            else:
                sphere = generate_sphere(radius)
                if not spheres_overlap(sphere):
                    spheres.append(sphere)
    return spheres

def simulate_ball_stack(n, box_size, radius):
    def distance_between_balls(ball1, ball2):
        return np.linalg.norm(ball1['position'] - ball2['position'])
    # 设置盒子和球的参数
    box_dimensions = np.array(box_size)
    ball_radius = radius
    num_balls = n

    # 初始化球的状态
    balls = [{'position': np.random.rand(3) * box_dimensions, 'velocity': np.zeros(3)}]

    # 模拟过程
    time_step = 0.01
    gravity = np.array([0, 0, -9.8])
    iteration = 0
    while len(balls)< num_balls:
        if iteration % 1000 == 0:
            print('计算中{}'.format(iteration))
        iteration +=1
        # 更新球的位置和速度
        for ball in balls:
            ball['position'] += ball['velocity'] * time_step + 0.5 * gravity * time_step**2
            ball['velocity'] += gravity * time_step

            # 反弹检查
            for i in range(3):
                if ball['position'][i] - ball_radius < 0:
                    ball['position'][i] = ball_radius
                    ball['velocity'][i] = abs(ball['velocity'][i]) * 0.5

                if ball['position'][i] + ball_radius > box_dimensions[i]:
                    ball['position'][i] = box_dimensions[i] - ball_radius
                    ball['velocity'][i] = -abs(ball['velocity'][i]) * 0.5

        # 碰撞检查
        if len(balls) <= 1:
            pass
        else:
            for i in range(len(balls)):
                for j in range(len(balls)):
                    dist = np.linalg.norm(balls[i]['position'] - balls[j]['position'])
                    if dist < 2 * ball_radius:
                        relative_velocity = balls[i]['velocity'] - balls[j]['velocity']
                        normal = (balls[i]['position'] - balls[j]['position']) / dist

                        balls[i]['velocity'] -= 1 * np.dot(relative_velocity, normal) * normal
                        balls[j]['velocity'] += 1 * np.dot(relative_velocity, normal) * normal

        # 生成新球
        if len(balls) < num_balls:
            new_ball = {'position': np.random.rand(3) * box_dimensions, 'velocity': np.zeros(3)}
            overlapping = any(distance_between_balls(new_ball, existing_ball) < 2 * ball_radius for existing_ball in balls)
            if not overlapping:
                balls.append(new_ball)

        # 判断结束条件
        total_velocity_change = np.sum([np.linalg.norm(ball['velocity']) for ball in balls])
        if total_velocity_change < 0.1 and len(balls) == num_balls:
            break

    # 输出结果
    result = [(ball['position'][0], ball['position'][1], ball['position'][2], ball_radius) for ball in balls]
    return result


data_file_list = os.listdir(r'/Volumes/新加卷/硕士毕业设计-宋梓贤/颗粒种子/8-particle-sinter/output_data')
main_dir = '/Volumes/新加卷/硕士毕业设计-宋梓贤/颗粒种子/8-particle-sinter/'
width = 150
box_size = [width,width,width]
radius = 26
number = 7
sample_num = 10
sintering_temperature = [300]#,400,500,600,700,800,900,1000] 
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
        #spheres = generate_tangent_spheres(number, x_min, x_max, y_min, y_max, z_min, z_max)
        spheres = simulate_ball_stack(number, box_size,radius)
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
            file.write(f"region box block -{width/2} {width/2} -{width/2} {width/2} -{width/2} {width/2} units box\n")
            file.write("create_box 2 box\n")
            for index, data in enumerate(spheres, start=1):
                file.write(f"read_data random_data_${{r{index}}}.data add append shift {data[0]-width/2} {data[1]-width/2} {data[2]-width/2}\n")
            file.write("#------------------------------开始模拟\n")
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
            file.write("pair_style eam/alloy\n")
            file.write("pair_coeff * * Cu_mishin1.eam.alloy Cu Cu\n")
            file.write("fix allmo all momentum 1 linear 1 1 1 angular\n")
            file.write("minimize 1.0e-4 1.0e-6 100 1000\n")
            file.write(f"fix mynpt all npt temp 300 300 0.1 iso 0 0 1\n")
            file.write("run 100000\n")
            file.write("unfix mynpt\n")
            file.write(f"fix mynpt all npt temp 300 {sintering_temperature} 0.1 iso 0 0 1\n")
            #file.write(f"shell mkdir {sintering_temperature}\n")
            if sintering_temperature == 300:
                run_time = 10000
            else:
                run_time = (sintering_temperature-300)*1000
            file.write(f"run {run_time}\n")
            file.write("unfix mynpt\n")
            file.write(f"fix mynpt all npt temp {sintering_temperature} {sintering_temperature} 0.1 iso 0 0 1\n")
            file.write(f"dump a all custom 10000 *.atom id type x y z\n")
            file.write("run 3500000\n")
            file.write("#------------------------------冷却得到模型\n")
            file.write("unfix mynpt\n")
            file.write(f"fix mynpt all npt temp {sintering_temperature} 300 0.1 iso 0 0 1\n")
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

 