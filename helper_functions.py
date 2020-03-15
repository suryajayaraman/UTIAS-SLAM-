#!/usr/bin/env python
# coding: utf-8

import numpy as np


#default values for reading data
data_folder_path = '../MRCLAM_Dataset1/'
Landmarks_GT_filename = 'Landmark_Groundtruth.dat'
barcocdes_filename = 'Barcodes.dat'


def load_MRCLAM_dataSet(dataset_path = data_folder_path, n_robots = 1):
    """
    function for loadin MRCLAM dataset
    
    Input
    -----
    dataset_path - path to load data from
    n_robots     - no of robots for which data is to be read
    
    Output
    ------
    Robots - A dict of keys = n_robots; each key is paired to another dict with ground truth pose data,
    # odometry data and measurement data
    """
    
    print('Loading data')
    try:
        # Barcodes dat format
        # Subject #Barcode 
        barcoders_data     = np.loadtxt(data_folder_path + barcocdes_filename)

    except:
        print('Error in Reading barcode data')
        barcoders_data = None

    try:
        # Landmark Groundtruth Data Fomat:
        # Subject #    x [m]    y [m]    x std-dev [m]    y std-dev [m] 
        Landmarks_GT_data     = np.loadtxt(data_folder_path + Landmarks_GT_filename)
    except:
        print('Error in Reading Landmarks_GT_data')
        Landmarks_GT_data = None
        
    Robots = {}
    
    for i in range(1, n_robots+1):
        try:
            print('Reading robot' + str(i) + '_Groundtruth data')
            robot_pose_data = np.loadtxt(data_folder_path + 'Robot' + str(i) + '_Groundtruth.dat')
            pose_status = 1
        except: 
            print('Error in Reading robot' + str(i) + '_Groundtruth data')
            pose_status = 0

        try:
            print('Reading robot' + str(i) + '_odometry data')
            robot_odom_data = np.loadtxt(data_folder_path + 'Robot' + str(i) + '_Odometry.dat')
            odom_status = 1
        except:
            print('Error in Reading robot' + str(i) + '_odometry data')
            odom_status = 0
            
        try:
            print('Reading robot' + str(i) + '_measurements data')
            robot_measurement_data = np.loadtxt(data_folder_path + 'Robot' + str(i) + '_Measurement.dat')
            measurment_status = 1
        except:
            print('Error in Reading robot' + str(i) + '_measurements data')
            measurment_status = 0
            
        if (pose_status and odom_status and measurment_status):
            robot_data = {'G' : robot_pose_data, 'O' : robot_odom_data, 'M' : robot_measurement_data}
            Robots[str(i)] = robot_data        
            print('Robot' + str(i) + ' data read successfully')
            print('------------------------------------------------')
    print('All data successfully read')
    print('------------------------------------------------')
    return barcoders_data, Landmarks_GT_data, Robots


def conBear(oldBear):
    """
    function to constrain input angle to -pi to pi
    
    Input
    -----
    Angle in radians
    
    output
    -----
    Angle in radians (in range -pi to pi) 
    """
    while oldBear < -np.pi:
        oldBear = oldBear + 2* np.pi
    
    while oldBear > np.pi:
        oldBear = oldBear - 2* np.pi
    newBear = oldBear;
    
    return newBear



def getObservations(Robots, robot_num, t, index, codeDict):
    """
    Gives the list of measurements for the input robot around the timestamp t

    Input
    -----
    Robots     - dictionary containing the relevant robot information including Groundtruth pose and the Estimated pose
    robot_num  - index of the robot of interest
    t          - timestamp (in seconds) of interest for getting measurements
    index      - index of measurement data available, from the prevous step
    codeDict   - A dict containing the Mapping of Barcode numbers to subject ID
    
    Output
    ------
    z          - 2D array of shape (n x 3) where n is no of measurements relevant to input timestamp and each containing
                 range, bearing and landmarkID 
    index      - updated measurement index of the latest measurement
    """
    
    z = np.zeros((3,1))
    if (index <= Robots[str(robot_num)]['M'].shape[0]-1):
        while (Robots[str(robot_num)]['M'][index, 0] - t < .005) and (index <= Robots[str(robot_num)]['M'].shape[0]-1):
            barcode = int(Robots[str(robot_num)]['M'][index,1])
            landmarkID = 0;
            
            if (barcode in codeDict):
                landmarkID = codeDict[barcode]
            else:
                print('key not found')
                
            if landmarkID > 5 and landmarkID < 21:
                r     = Robots[str(robot_num)]['M'][index, 2];
                theta = Robots[str(robot_num)]['M'][index, 3];

                #if its the first measurement found
                if int(z[2,0]) == 0:
                    z = np.array([r, theta, landmarkID - 5]).reshape(3,1)
                else:
                    newZ = np.array([r, theta, landmarkID - 5]).reshape(3,1)
                    z = np.column_stack((z, newZ))

            index = index + 1;
            if index > Robots[str(robot_num)]['M'].shape[0]-1:
                break
    return z, index


def path_loss(Robots, robot_num, start):
    """
    computes euclidean loss between robot's estimated path and ground truth
    ignores bearing error
    
    Input
    -----
    Robots - dictionary containing the relevant robot information including Groundtruth pose and the Estimated pose
    start  - starting index from which to calculate the eucliden loss
    
    Output
    ------
    loss   - euclidean difference in position between the Groundtruth pose and the Estimated pose
    """
    loss = 0
    for i in range(start, Robots[str(robot_num)]['G'].shape[0]):
        x_diff = Robots[str(robot_num)]['G'][i,1] - Robots[str(robot_num)]['Est'][i,1];
        y_diff = Robots[str(robot_num)]['G'][i,2] - Robots[str(robot_num)]['Est'][i,2];
        err = np.power(x_diff,2) + np.power(y_diff, 2)
        loss = loss + err;
    return loss


def sample_MRCLAM_dataSet(Robots, sample_time =0.02):
    """
    UTIAS Multi-Robot Cooperative Localization and Mapping Dataset produced by Keith Leung (keith.leung@robotics.utias.utoronto.ca) 2009

    Samples the dataset at fixed intervals. Odometry data is interpolated using the recorded time.
    Measurements are rounded to the nearest timestep. 
    
    Input
    -----
    Robots      -
    sample_time - constant time stamp to sample from, in seconds (default is 0.02s)


    Output
    ------
    Robots    - Dict with updated Groundtruth, odometry and measurement data (constant sample time)
    timesteps - No of timestamps of input sample time  
    """
    
    n_robots = len(Robots.keys())

    min_time = Robots['1']['G'][0,0]
    max_time = Robots['1']['G'][-1,0]

    #finding the min timestamp across all the robots
    for n in range(2, n_robots + 1):
        min_time = min(min_time, Robots[str(n)]['G'][0,0])
        max_time = max(max_time, Robots[str(n)]['G'][-1,0])

    #subtracting min timestamp from all variables to make it realtive to min_time
    for n in range(1, n_robots + 1):
        Robots[str(n)]['G'][:,0] = Robots[str(n)]['G'][:,0] - min_time
        Robots[str(n)]['M'][:,0] = Robots[str(n)]['M'][:,0] - min_time
        Robots[str(n)]['O'][:,0] = Robots[str(n)]['O'][:,0] - min_time

    max_time = max_time - min_time;
    timesteps = int(np.floor(max_time/sample_time)+1)

    #iterate for every robot
    for n in range(1, n_robots + 1):
        #get old data (original series of data)
        oldData = Robots[str(n)]['G']

        k = -1;
        t = 0;
        i = 0;
        p = 0;

        nr,nc = oldData.shape
        newData = np.zeros((timesteps,nc))

        while(t <= max_time):
            newData[k+1,0] = t

            #incrementing i till the timestep till 
            #which current sample_time is less than or equal to
            while(oldData[i,0] <= t):        
                if(i==nr-1):
                    break
                i = i + 1
            if( (i==0) or (i == (nr-1)) ):
                newData[k+1, 1:] = 0
            else:
                p = (t - oldData[i-1,0]) / (oldData[i,0] - oldData[i-1,0])

                #if the no of columns is 8 then start with 3rd column
                if(nc == 8):
                    sc = 2
                    newData[k+1,1] = oldData[i,1] #keep id number

                #else start with 2nd column
                else:
                    sc = 1

                #for each column other starting with sc
                #find newdata corresponding to olddata using gradient 
                for c in range(sc, nc):
                    if( (nc==8) and (c>=6) ):
                        d = oldData[i,c] - oldData[i-1,c];
                        if d > np.pi:
                            d = d - 2*pi;
                        elif d < -np.pi:
                            d = d + 2*pi;
                        newData[k+1,c] = p*d + oldData[i-1,c];
                    else:
                        newData[k+1,c] = p*(oldData[i,c] - oldData[i-1,c]) + oldData[i-1,c];
            k = k + 1
            t = t + sample_time

    #         if int(t) % 100 == 0:
    #             print(str(t) + 'timestamps done')

        Robots[str(n)]['G'] = newData
    
    
        oldData = Robots[str(n)]['O']
        k = -1;
        t = 0;
        i = 0;
        p = 0;

        nr,nc = oldData.shape
        newData = np.zeros((timesteps,nc))

        while(t <= max_time):
            newData[k+1,0] = t

            #incrementing i till the timestep till 
            #which current sample_time is less than or equal to
            while(oldData[i,0] <= t):        
                if(i==nr-1):
                    break
                i = i + 1
            if( (i==0) or (i == (nr-1)) ):
                newData[k+1, 1:] = oldData[i,1:]
            else:
                p = (t - oldData[i-1,0]) / (oldData[i,0] - oldData[i-1,0])

                #if the no of columns is 8 then start with 3rd column
                if(nc == 8):
                    sc = 2
                    newData[k+1,1] = oldData[i,1] #keep id number

                #else start with 2nd column
                else:
                    sc = 1

                #for each column other starting with sc
                #find newdata corresponding to olddata using gradient 
                for c in range(sc, nc):
                    if( (nc==8) and (c>=6) ):
                        d = oldData[i,c] - oldData[i-1,c];
                        if d > np.pi:
                            d = d - 2*pi;
                        elif d < -np.pi:
                            d = d + 2*pi;
                        newData[k+1,c] = p*d + oldData[i-1,c];
                    else:
                        newData[k+1,c] = p*(oldData[i,c] - oldData[i-1,c]) + oldData[i-1,c];
            k = k + 1
            t = t + sample_time
 
        Robots[str(n)]['O'] = newData
    
        oldData = Robots[str(n)]['M']
        newData=oldData    
        for i in range(0, oldData.shape[0]):
            newData[i,0] = np.floor(oldData[i,0]/sample_time + 0.5)*sample_time; 
        Robots[str(n)]['M'] = newData
    
    return Robots, timesteps


"""
command to load data of n robots 
"""
##barcoders_data, Landmarks_GT_data, Robots = load_MRCLAM_dataSet(n_robots)


"""
command to sample to data at constant sample time
"""