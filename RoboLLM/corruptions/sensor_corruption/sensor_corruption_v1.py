@TRANSFORMS.register_module()
class sensor_corruption(BaseTransform):
    def __init__(self,model_type=None,corruption_level=None):
        self.model_type=model_type
        self.corruption_level=corruption_level
    def transform(self, results):
        if self.model_type=='MMDrive':
            if self.corruption_level=='low':
                results['v0']=results['v0']*0.5
                results['acc_x']=results['acc_x']*0.5
                results['acc_y']=results['acc_y']*0.5
                results['steering']=random.uniform(-7.7,6.3)
            elif self.corruption_level=='medium':
                results['v0']=results['v0']*3
                results['acc_x']=results['acc_x']*3
                results['acc_y']=results['acc_y']*3
                results['steering']=random.uniform(-7.7,6.3)
            elif self.corruption_level=='high':
                results['v0']=results['v0']*6
                results['acc_x']=results['acc_x']*6
                results['acc_y']=results['acc_y']*6
                results['steering']=random.uniform(-7.7,6.3)
            coefficient_matrix = np.array([
                            [Sx, Cxy, Cxz],
                            [Cyx, Sy, Cyz],
                            [Czx, Czy, Sz]
                        ]) 
            noise_vector = np.random.normal(mean, std_dev, 3)
            bias_vector = np.array([bias_1, bias_2, bias_3])
            accel_c=np.dot(accel,coefficient_matrix) + bias_vector + noise_vector
            ax_c,ay_c=accel_c[:2]
            ax,ay=accel[:2]
            velocity_c = velocity[0]+math.sqrt((ax_c-ax)**2+(ay_c-ay)**2)
            steering_c = steering+np.random.normal(mean, std_dev, 1)
            if steering_c>6.3:
                steering_c=6.3
            elif steering_c<-7.7:
                steering_c=-7.7
        elif self.model_type=='drivevlm':
            question=results['questions'][3]
            traj=question.split("Historical Trajectory (last 2 seconds): ")[1].split("\n")[0]
            traj = ast.literal_eval(traj)
            traj = np.array(traj)
            traj=np.append(traj,[0.0,0.0])
            traj = traj.reshape(-1, 2)
            delta = np.diff(traj, axis=0)
            t = 0.5
            accel = (2 * delta) / t**2
            zeros = np.zeros((accel.shape[0], 1))  # 创建一个形状为 (n, 1) 的零矩阵
            accel = np.hstack([accel, zeros]) 
            Sx, Sy, Sz = 1, 1, 1  # 这里先假设系数矩阵中的对角元素，可根据实际修改
            Cxy, Cxz, Cyx, Cyz, Czx, Czy = 0, 0, 0, 0 ,0,0 # 非对角元素，可根据实际修改
            coefficient_matrix = np.array([
                [Sx, Cxy, Cxz],
                [Cyx, Sy, Cyz],
                [Czx, Czy, Sz]
            ])
            mean = 0  
            std_dev = 1  
            noise_vector = np.random.normal(mean, std_dev, 3)
            bias_vector = np.array([0.3, 0.3, 0.2])
            accel_c=np.dot(accel,coefficient_matrix) + bias_vector + noise_vector
            accel_c=accel_c[:,:-1]
            move=accel_c*t**2/2
            end_point = np.array([0, 0])
            corrupution_trajectory = [end_point]
            for delta in reversed(move):
                prev_point = corrupution_trajectory[-1] - delta
                corrupution_trajectory.append(prev_point)
            corrupution_trajectory = np.array(corrupution_trajectory[::-1])
            corrupution_trajectory = corrupution_trajectory[:-1]
            formatted_traj = ", ".join([f"({x:.2f},{y:.2f})" for x, y in corrupution_trajectory])
            new_traj_str = f"[{formatted_traj}]"
            prefix, suffix = question.split("Historical Trajectory (last 2 seconds): ")
            new_text = prefix + f"Historical Trajectory (last 2 seconds): {new_traj_str}\n" + suffix.split("\n", 1)[1]
            results['questions'][3]=new_text
        elif self.model_type=='openemma':
            if self.corruption_level=='low':
                results['sensor_corruption']=0.1
            elif self.corruption_level=='medium':
                results['sensor_corruption']=0.2
            elif self.corruption_level=='high':
                results['sensor_corruption']=0.3
        return results



def corruption_trajectory(traj, t=0.5, Sx=1, Sy=1, Sz=1, Cxy=0, Cxz=0, Cyx=0, Cyz=0, Czx=0, Czy=0, mean=0, std_dev=1, bias_vector=np.array([0.3, 0.3, 0.2])):
    traj = np.array(traj)

    # 计算位移
    delta = np.diff(traj, axis=0)

    # 计算加速度
    accel = (2 * delta) / t**2

    # 定义系数矩阵
    coefficient_matrix = np.array([
        [Sx, Cxy, Cxz],
        [Cyx, Sy, Cyz],
        [Czx, Czy, Sz]
    ])

    # 生成噪声向量
    noise_vector = np.random.normal(mean, std_dev, 3)

    # 修改加速度
    accel_c = np.dot(accel, coefficient_matrix) + bias_vector + noise_vector
    accel_c[:, -1] = 0  # 将最后一个维度（z 方向）设置为 0

    # 计算修改后的位移
    move = accel_c * t**2 / 2

    # 反推出修改后的轨迹点
    end_point = traj[-1]
    corruption_trajectory = [end_point]
    for delta in reversed(move):
        prev_point = corruption_trajectory[-1] - delta
        corruption_trajectory.append(prev_point)

    # 反转轨迹点列表并转换为列表
    corruption_trajectory = np.array(corruption_trajectory[::-1]).tolist()

    return corruption_trajectory