import os
import time
import pickle
import numpy as np
import struct
from read_params import read_params

class tracer_file():
    ''''''
    def __init__(self,path,extended_output=True,Nspecies=5):
        self.path = path
        self.snappath = path + "/output/"
        self.Nspecies = Nspecies
        self.extended_output = extended_output

        self.params = read_params(self.snappath)
        self.path_file1 = os.path.join(self.snappath,self.params["TracerOutputFile"])
        self.path_file2 = os.path.join(self.snappath,"tracer.dat") # Hard-coded
        path_conf = os.path.join(path,self.params["TracerOutputConfFile"])

        file1  = open(self.path_file1, "rb")
        # Reading header
        file1.seek(4, 0)  # Skip dummy bytes
        self.NTracer, = struct.unpack("i", file1.read(4))
        file1.seek( 8, 1 ) # Skip 8 dummy bytes
        self.Masses = np.fromfile(file1, count=self.NTracer, dtype='f8' )
        file1.close()
        self.tracer_dt = np.genfromtxt(path_conf,skip_header=1)[1] # Hacky way of getting dt

        if self.extended_output:
            # 4 [dummy] + 8 [time] + 4*(10 + Nspecies) [vals] + 4 [dummy]
            self.bts_per_output = 16 + self.NTracer * (10 + self.Nspecies) * 4
        else:
            # 4 [dummy] + 8 [time] + 4*6 [vals] + 4[dummy]
            self.bts_per_output = 16 + self.NTracer * 6 * 4

        # If there is only one tracers file this should do the trick
        if self.path_file1 == self.path_file2:
            self.offset = 20 + self.NTracer * 8
        else:
            self.offset = 0

        self.data = {}
        self.data["id"] = np.arange(self.NTracer) # Probably set at a differen part of the code
        self.NTimesteps = 0 # Initialize value
        self.vals_dict = {
            "pos" : [1,2,3],
            "posx" : 1,
            "posy" : 2,
            "posz" : 3,
            "time" : 0,
            "rho" : 4,
            "temp" : 5,
            "u" : 6,
            "vel" : [7,8,9],
            "Dedt" : 10,
            "xnuc" : [sp + 11 for sp in range(self.Nspecies)]
        }


    def __getattr__(self,name):
        """enable access via object attributes to data dict entries"""
        if name in self.data:
            return self.data[name]
        raise AttributeError("Class tracer_file has no attribute '{}'.".format(name))

    def find_tracers_at_time(self,time):
        '''Returns tracer information only at specific time
           should (hopefully) be very fast'''
        index = round(time/self.tracer_dt)
        file2 = open(self.path_file2, "rb")
        file2.seek(self.offset,0) # Go to start of file
        file2.seek(index*self.bts_per_output,1) # Go to desired output
        vals = self.read_vals(file2)
        file2.close()
        self.data["time"] = vals[0]
        self.write_vals_to_data(vals)
        return self.data.copy()

    def find_tracers_in_box(self,box,center):
        '''Returns ids of tracers contained in a box of size "box" centered on "center"

        Parameters
        ----------
        box : float or list of floats
            dimensions of the box in cm

        center : float or list of floats
            position of the center in cm



           '''
        if type(box) == int or type(box)==float:
            box = np.array([box,box,box])
        if type(center) == int or type(center)==float:
            center = np.array([center,center,center])
        rel_pos = np.abs(self.data["pos"] - center)
        cond_x = (np.abs(rel_pos[:,0])<box[0])
        cond_y = (np.abs(rel_pos[:,1])<box[1])
        cond_z = (np.abs(rel_pos[:,2])<box[2])
        return self.data["id"][np.logical_and(np.logical_and(cond_x,cond_y),cond_z)]

    def save_followed_tracers(self,filename="followed_tracers",n=-1,path=-1):
        if n==-1:
            n = 0
        save_name = filename + str(n).zfill(2)
        if path == -1:
            path = './'
        if self.followed_tracers != 0:
            if os.path.isfile(os.path.join(path,save_name)):
                while os.path.isfile(os.path.join(path,save_name)):
                    n+=1
                    print("File {:s} already existing, saving as {:s}".format(save_name,filename+str(n).zfill(2)))
                    save_name = filename + str(n).zfill(2)
                with open(os.path.join(path,save_name), 'wb') as f:
                    pickle.dump(self.followed_tracers, f)
            else:
                with open(os.path.join(path,save_name), 'wb') as f:
                    pickle.dump(self.followed_tracers, f)
    def read_followed_tracers(self,filename="followed_tracers",n=-1,path=-1):                    
        if n==-1:
            n = 0
        save_name = filename + str(n).zfill(2)
        if path == -1:
            path = './'
        print('Reading {:s}'.format(os.path.join(path,save_name)))
        with open(os.path.join(path,save_name), 'rb') as f:
            self.followed_tracers = pickle.load(f)

    def read_vals(self,f):
        '''Reads and returns values from tracer file
        f should be pointing at the beginning of an output
        Outputs are shaped:
            4 [dummy] + 8 [time] + 4*6 [vals] + 4[dummy]
        unless extended_output is enabled, in that case they are:
            4 [dummy] + 8 [time] + 4*(10 + Nspecies) [vals] + 4 [dummy]
        '''
        sp = []
        f.seek(4,1) # Skip dummy
        t, = struct.unpack("d", f.read(8)) # Time is double (8 bytes)
        x = np.fromfile(f, count=self.NTracer, dtype='f4')
        y = np.fromfile(f, count=self.NTracer, dtype='f4')
        z = np.fromfile(f, count=self.NTracer, dtype='f4')
        rho = np.fromfile(f, count=self.NTracer, dtype='f4')
        temp = np.fromfile(f, count=self.NTracer, dtype='f4')
        u = np.fromfile(f, count=self.NTracer, dtype='f4')
        if self.extended_output:
            vx = np.fromfile(f, count=self.NTracer, dtype='f4')
            vy = np.fromfile(f, count=self.NTracer, dtype='f4')
            vz = np.fromfile(f, count=self.NTracer, dtype='f4')
            Dedt = np.fromfile(f, count=self.NTracer, dtype='f4')
            for i in range(self.Nspecies):
                sp.append(np.fromfile(f, count=self.NTracer, dtype='f4'))
        f.seek(4,1)
        if self.extended_output:
            return t, x, y, z, rho, temp, u, vx, vy, vz, Dedt, sp
        else:
            return t, x, y, z, rho, temp, u

    def write_vals_to_data(self,vals):
        '''Stores the read values into data dictionary'''
        self.data["pos"] = np.array([vals[1],vals[2],vals[3]]).T
        self.data["rho"] = vals[4]
        self.data["temp"] = vals[5]
        self.data["u"] = vals[6]
        if self.extended_output:
            self.data["vel"] = np.array([vals[7],vals[8],vals[9]]).T
            self.data["Dedt"] = vals[10]
            self.data["xnuc"] = np.asarray(vals[11]).T
        return

    def flatten_list(self,l):
        '''Terrible function that I built bc I seem to be incapable
        of flattening a list shaped like [int,[int,int,int]]'''
        l_flat = []
        for i in l:
            if type(i)==int:
                l_flat.append(i)
            if type(i)==list:
                for k in i:
                    if type(k) != int:
                        print("WTF, I give up")
                        raise
                    l_flat.append(k)
        return l_flat

    def byte_map(self,time_index,val_index):
        '''returns position in binary file of a piece of data.
            should not be used with val_index = 0 (time)
            as it assumes the pointer is pointing right after
            where time is stored'''
        # 4 [dummy] + 8 [time] + 4*(10 + Nspecies) [vals] + 4 [dummy]
        return self.offset+time_index*self.bts_per_output+12+(val_index-1)*4*self.NTracer

    def byte_map_per_line(self,val_index):
        '''returns number of bytes after the position where "time" is stored
        in the binary, for a certain magnitude, i.e:
        val    == val_index = n bytes
        "posx" == 1   = 0
        "posy" == 2   = 4 * NTracer
        "rho"  == 4   = 3 * 4 * Ntracer
        '''
        # 4 [dummy] + 8 [time] + 4*(10 + Nspecies) [vals] + 4 [dummy]
        return (val_index-1)*4*self.NTracer


    def get_values_at_times(self,values,times,ids=-1):
        '''This allows to only read certain values of tracers
        at certain times. It can be helpfull if you have too many
        tracers and you cannot load all the information of the tracers
        because of memory concerns. Otherwise reading all the info of
        timesteps with "find_tracers_at_time" should be faster'''
        if type(values) == str:
            values = [values]
        if not hasattr(times, '__iter__'):
            times = [times]

        val_indices = [self.vals_dict[val] for val in values]
        val_indices = self.flatten_list(val_indices)
        time_indices = [round(i) for i in times/self.tracer_dt]
        self.data_at = {}
        file2 = open(self.path_file2, "rb")
        for t_index in time_indices:
            file2.seek(t_index*self.bts_per_output+self.offset,0)
            t = struct.unpack("d", file2.read(8))
            val_data = {}
            val_data["time"] = t
            for val_index in val_indices:
                file2.seek(self.byte_map(t_index,val_index),0)
                val_data[val_index] = np.fromfile(file2, count=self.NTracer, dtype='f4')
            for val in values:
                if val == "pos":
                    val_data[val] = np.array([val_data[1],val_data[2],val_data[3]]).T
                elif val == "vel":
                    val_data[val] = np.array([val_data[7],val_data[8],val_data[9]]).T
                elif val == "xnuc":
                    val_data[val] = np.asarray([val_data[sp] for sp in self.vals_dict["xnuc"]]).T
                else:
                    val_data[val] = val_data[self.vals_dict[val]]
            for val_index in val_indices:
                val_data.pop(val_index, None)
            self.data_at[t_index] = val_data
        file2.close()
        return

    def read_single_tracer(tracer_id,file,val_indices):
        pos_per_line = self.byte_map()
        line = 0
        list_of_indices_to_read = np.array([self.byte_map_per_line(i)+tracer_id for i in val_indices])

        while len(file.read(1)>0):
            file.seek(line*self.bts_per_output+self.offset+12,0)
            for i in val_indices:
                file.seek(self.bts_per_output+self.offset+4,0)


    def follow_tracers(self,ids,vals=["pos"],guess_tps=1001):
        '''Returns a dictionary with the information about the specified
        tracers over the entire simulation time

        Parameters
        ----------
        ids : int or list of ints
            ids of the tracers to follow
        vals : str or list of str, optional
            Type of data to load. By default loads position only
        guess_tps : int, optional
            Guess for the total number of timesteps, if unknown

        Returns
        -------
        tracer : dict of dict
            Dictionary with the ids of the tracers as the first key and
            vals as the second key.
            The value returned by the second key is an array with lenght
            equal to the number of timesteps in the simulation, it will be
            1-Dimensional if val is 1-D with len=NTimesteps
            for "pos", "vel", and "xnuc" it will have extra dimensions

        Examples
        --------
        >>> tracer = tracer_file.follow_tracers([0, 1],vals=["pos","temp"])
        >>> tracer[0]["pos"].shape()
        (200, 3)
        >>> tracer[0]["temp"].shape()
        (200,)
        '''
        tracer = {}
        file = open(self.path_file2, "rb")
        line = 0
        skipped = 0  # Lines skipped in case there is overlapping data
        start = time.time()
        times = [0] # We need it not empy for the check for overlapping data
        # Make sure we can iterate over vals and ids
        if type(vals) == str:
            vals = [vals]
        if type(ids) == int:
            ids = [ids]

        # Make a single flat list of vals
        val_indices = [self.vals_dict[val] for val in vals]
        val_indices = np.asarray(self.flatten_list(val_indices))

        # Compute the positions in the timestep that we will read
        indices_to_read = []
        for tracer_id in ids:
            for i in val_indices:
                indices_to_read.append(self.byte_map_per_line(i)+tracer_id*4)
        indices_to_read = np.sort(np.asarray(indices_to_read))
        # Compute bytes to move after each read
        rel_indices_to_read = np.diff(indices_to_read,prepend=0)
        # Substract 4bytes because when we DO read we move the pointer 4 bytes
        # Except for the first value to read
        rel_indices_to_read[1:] = rel_indices_to_read[1:]-4

        # Create array for storing data
        if self.NTimesteps != 0:
            temp_array = np.zeros((self.NTimesteps,len(ids)*len(val_indices)))
        else:  # We estimate NTimestep
            temp_array = np.zeros((guess_tps,len(ids)*len(val_indices)))
        while len(file.read(1)) > 0 :
            #  If we underestimate NTimestep create new array twice as large
            if temp_array.shape[0] <= line-skipped:
                temp_ = np.zeros((temp_array.shape[0]*2,len(ids)*len(val_indices)))
                temp_[:temp_array.shape[0],:] = temp_array
                temp_array = temp_
                temp_ = 0
            file.seek(line*self.bts_per_output+self.offset+4,0)
            t = struct.unpack("d", file.read(8))[0]
            # Check for overlapping data
            if t <= times[-1]:
                skipped+=1
                line+=1
                continue
            times.append(t)
            for j, index in enumerate(rel_indices_to_read):
                file.seek(index,1)
                temp_array[line-skipped,j] = struct.unpack("f",file.read(4))[0]
            if line % 100 ==0:
                print("We are at line {:d}, time={:f}".format(line,t))
            file.seek((line+1)*self.bts_per_output+self.offset,0)
            line+=1
        times.pop(0)  # Remove the extra [0] that we added at the beginning
        self.NTimesteps = line-skipped

        print("Done in {:f}".format(time.time()-start))
        print("We have skipped {:d} lines of duplicated data".format(skipped))
        print("Reshaping the data array")
        for j, tracer_id in enumerate(ids):
            tracer[tracer_id]={"time":np.asarray(times)}
            for val in vals:
                if val == "pos":
                    tracer[tracer_id]["pos"] = np.array([temp_array[:,np.where(val_indices==1)[0][0]*len(ids)+j],temp_array[:,np.where(val_indices==2)[0][0]*len(ids)+j],temp_array[:,np.where(val_indices==3)[0][0]*len(ids)+j]]).T
                    tracer[tracer_id][val] = tracer[tracer_id][val][:self.NTimesteps,:]

                elif val == "vel":
                    tracer[tracer_id]["vel"] = np.array([temp_array[:,np.where(val_indices==7)[0][0]*len(ids)+j],temp_array[:,np.where(val_indices==8)[0][0]*len(ids)+j],temp_array[:,np.where(val_indices==9)[0][0]*len(ids)+j]]).T
                    tracer[tracer_id][val] = tracer[tracer_id][val][:self.NTimesteps,:]

                elif val == "xnuc":
                    tracer[tracer_id]["xnuc"] = np.asarray([temp_array[:,np.where(val_indices==sp)[0][0]*len(ids)+j] for sp in self.vals_dict["xnuc"]]).T
                    tracer[tracer_id][val] = tracer[tracer_id][val][:self.NTimesteps,:]

                else:
                    tracer[tracer_id][val] = temp_array[:,np.where(val_indices==self.vals_dict[val])[0][0]*len(ids)+j].T
                    tracer[tracer_id][val] = tracer[tracer_id][val][:self.NTimesteps]

        file.close()
        self.followed_tracers = tracer
        return tracer
    
    def access_val(self, val):
        '''Allows vectorized acess to followed tracers data
        returns 3D array of shape
        (len(ids),NTimesteps,len(val))
        '''
        array_of_dicts = np.array([self.followed_tracers[key] for key in self.followed_tracers.keys()])
        return np.asarray([i[val] for i in array_of_dicts])

    def add_followed_tracers(self,filename="followed_tracers",n=-1,path=-1):
        '''
        Combines currently followed_tracers (either loaded or computed)
        with an already existing followed_tracers file.
        Can be used to combine different followed_tracers files
        '''
        if n==-1:
            n = 0
        save_name = filename + str(n).zfill(2)
        if path == -1:
            path = './'
        with open(os.path.join(path,save_name), 'rb') as f:
            old_followed_tracers = pickle.load(f) 
        # Check for repeating tracer ids to drop them out
        old_ids = np.array([ids for ids in old_followed_tracers.keys()])
        new_ids = np.array([ids for ids in self.followed_tracers.keys()])
        ids_to_add = old_ids[np.isin(old_ids,new_ids,invert=True)]
        print("We have {:d} tracers that are in both datasets".format(len(old_ids)-len(ids_to_add)))
        start = time.time()
        for tracer_id in ids_to_add:
            self.followed_tracers[tracer_id] = old_followed_tracers[tracer_id]
        print("Combined datasets in {:f}s".format(time.time()-start))
