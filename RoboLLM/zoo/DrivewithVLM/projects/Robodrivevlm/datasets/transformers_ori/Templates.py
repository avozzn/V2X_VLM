
System_prompt =[
'''
**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle.

Inputs
1. Camera: Three pictures taken by the body camera from different angles, from the front <image>, front right <image>, front left <image>.
2. Lidar: Bird's Eye View point cloud image <image>, You're positioned at point (0,0) in image center point, facing the positive X-axis, Y-axis is perpendicular.
3. Radar: Bird's Eye View point cloud image <image>, Point length signifies velocity in the lidar's coordinate system.
4. Ego-States: Your current state including 
'''
]



System_prompt_only_ego=[
    '''
    Ego-States: Your current state including 
    '''
]
System_prompt_only_camera=[
    '''
    1.Camera: Three pictures taken by the body camera from different angles, from the front <image>, front right <image>, front left <image>.
    2.Ego-States: Your current state including
    '''
    
]
System_prompt_only_Lidar=[
    '''
    1.Lidar: Bird's Eye View point cloud image <image>, You're positioned at point (0,0) in image center point, facing the positive X-axis, Y-axis is perpendicular.
    2.Ego-States: Your current state including
    '''
]



System_prompt_no_image =[
'''
**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle.

Inputs
1. Lidar: Bird's Eye View point cloud image, You're positioned at point (0,0) in image center point, facing the positive X-axis, Y-axis is perpendicular.
2. Radar: Bird's Eye View point cloud image, Point length signifies velocity in the lidar's coordinate system.
3. Camera: Six pictures taken by the body camera from different angles, from the front, left front, right front, back, left back, right back.
4. Ego-States: Your current state including 
'''
]

System_prompt_no_camera=[
'''
**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle.

Inputs
1. Lidar: Bird's Eye View point cloud image <image>, You're positioned at point (0,0) in image center point, facing the positive X-axis, Y-axis is perpendicular.
2. Radar: Bird's Eye View point cloud image <image>, Point length signifies velocity in the lidar's coordinate system.
3. Ego-States: Your current state including 
'''

]

System_prompt_no_ego=[
'''
**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle.

Inputs
1. Camera: Three pictures taken by the body camera from different angles, from the front <image>, front right <image>, front left <image>.
2. Lidar: Bird's Eye View point cloud image <image>, You're positioned at point (0,0) in image center point, facing the positive X-axis, Y-axis is perpendicular.
3. Radar: Bird's Eye View point cloud image <image>, Point length signifies velocity in the lidar's coordinate system.
'''
  
]

System_prompt_no_text=[
'''
<image><image><image><image><image>
'''  
]

System_prompt_no_sensor =[
'''
**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle.

Inputs
1. You're positioned at point (0,0) in image center point, facing the positive X-axis, Y-axis is perpendicular.
2. Camera: Three pictures taken by the body camera from different angles, from the front <image>, left front <image>, right front <image>.
3. Ego-States: Your current state including 
'''
]


Scenario_Description=[
'''
Task
- Scene Description: Describe the current driving scenario, including the weather conditions, objects ahead that may impact driving, and critical traffic elements such as traffic lights.

'''
]





Instruction=[
'''
Instruction: 
- Primarily using LiDAR's positional data, combined with the semantic information from the camera's three perspectives and velocity information from the radar, perform the following inference.
'''
]


COT=[
'''
Based on the BEV Bounding Box you have given, proceed with the next inference steps.
'''
]

COT_2=[
'''
Based on the scenario description you have given, proceed with the next inference steps.
'''
]

VQA=[
'''
Task
- 
'''
]

###给全部的class token
Detection_BEV_Type1=[
'''
Task
- Object Detection: Identify the objects within the vehicle's front-facing area (up to approximately 10 meters) in the current driving scenario that belong to <class> and draw a Bird's Eye View bounding box around each one in the LiDAR point cloud image.

Output
- BEV Bounding Box:
 - [(x_center,x_size,y_center,y_size,yaw,category),...]
'''
]



###直接检测所有的类别(思维链，先检测有哪些类别，再根据类别检测物体)
Detection_BEV_Type2=[
'''


'''
]


Detection_3D_type1=[
'''



'''
]

Detection_3D_Type2=[
'''



'''
]

Segmentation_Type1=[
'''
'''
]


Segmentation_Type2=[
'''
'''
]

Planning = [
'''
Task
- Action Plan: Detail your meta-actions based on your analysis.
- Trajectory Planning: Develop a safe and feasible 3-second route using 6 new waypoints, one every 0.5 seconds.

Output
- Meta Action
- Trajectory (MOST IMPORTANT):
  - [(x1,y1), (x2,y2), ... , (x6,y6)]
'''
]