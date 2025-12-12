
###########################################################################
###########################################################################
###                                                                     ###
###        GALAXIES HOMEWORK 4: MODIFYING N-BODY SIMULATION CODE        ###
###                                                                     ###
###########################################################################
###########################################################################


###  I am going to take the original code that was in a jupyter notebook and      
###  paste it into here. I am doing this so it is easier to work with variables   
###  and time the code better                                                     

"""
Create Your Own N-body Simulation (With Python)
Adapted from Philip Mocz (2020) Princeton University, @PMocz
New structure, Initial condition module, movie module, rotational analysis, comments: C.Welker

Simulate orbits of particles interacting only through gravitational interactions.
The code calculates pairwise forces according to Newton's Law of Gravity. 
Note that there is no expansion yet in this simulation. We are focusing on a patch decoupled from
expansion. Let's see how long it takes to virialize once a structure is collapsing.
Of course in a real simulation with billions of particles, we will need some better approximation 
to NOT calculate all pairwise interactions (see optional exercise of Homework 4) 
But for now, let's focus on a few hundreds particles
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
import os
import moviepy.video.io.ImageSequenceClip
from natsort import natsorted
import time
import glob


############################################################################
############################################################################
###                                                                      ###
###            PASTING IN FUNCTIONS FROM THE JUPYTER NOTEBOOK            ###
###                                                                      ###
############################################################################
############################################################################


##---------------------------------------------------------------
##            Initializing the scale factor function:           -
##---------------------------------------------------------------

def ScaleFactor(t, t_end, comoving_distance, add_expansion = False):

    # I would like to explore the time in the Universe where 2 < z < 3. To do
    # this, I am going to use 2.143 Gyrs when z = 3 and 3.276 Gyrs when z = 2:
    t_redshift_start = 2.143 * u.Gyr
    t_redshift_end = 3.276 * u.Gyr
    
    # Because we are working in code units, we need to make sure that we 
    # adjust to Gyrs so the redshift and a values are correct and we are looking
    # over the right redshift range I chose above. We need to interpolate over
    # this range with the time steps we want:
    real_time = t_redshift_start + (t / t_end) * (t_redshift_end - t_redshift_start)
    redshift = z_at_value(Planck18.age, real_time)
    a = 1 / (1 + redshift)
    H = Planck18.H(redshift).to(u.Gyr**(-1)).value  # in 1/Gyr
    # Convert to code units (multiply by t_end since t is in code units)
    H_in_code_units = H * t_end
    
    if add_expansion == True:
        # Get proper distance:
        sim_distance = a * comoving_distance
    
    else:
        sim_distance = comoving_distance
    
    return a, H_in_code_units, sim_distance

##---------------------------------------------------------------
##          Defining the initial conditions function:           -
##---------------------------------------------------------------

def InitialConditions(N,
                      omega,
                      mtot,
                      v0, 
                      radial_velocity,
                      temp,
                      pure_radial_flow = False):
    
    # --------------This module generates Initial Conditions
    #For N particles with total mass mtot, solid angular velocity omega

    np.random.seed(17) # set the random number generator seed
    
    # Setting up the Maxwell-Boltzmann sigma: 
    mb_sigma = np.sqrt(temp)
    
    # Setting up the code so that it can take two random peaks or one:
    if len(N) > 1:
        # Establishing number of particles:
        N1 = N[0]
        N2 = N[1]
        
        # Finding mass and positions of each one:
        mass1 = mtot[0] * np.ones((N1, 1)) / N1
        pos1 = np.random.randn(N1, 3)
        
        mass2 = mtot[1] * np.ones((N2, 1)) / N2
        pos2 = np.random.randn(N2, 3)
        
        # Now adding in radial flow:
        if pure_radial_flow == True:
            magnitude1 = np.sqrt(np.sum(pos1**2, axis = 1, keepdims = True))
            magnitude2 = np.sqrt(np.sum(pos2**2, axis = 1, keepdims = True))
            
            vel_radial1 = radial_velocity[0] * (pos1 / magnitude1)
            vel_radial2 = radial_velocity[1] * (pos2 / magnitude2)
            
            random_vel1 = mb_sigma * np.random.randn(N1, 3)
            random_vel2 = mb_sigma * np.random.randn(N2, 3)
            
            vel1 = vel_radial1 + random_vel1
            vel2 = vel_radial2 + random_vel2
            
            vel1 -= np.mean(mass1 * vel1, 0) / np.mean(mass1)
            vel2 -= np.mean(mass2 * vel2, 0) / np.mean(mass2)
        
        elif(v0 == 1):
            
            ### Starting with the first halo:
            
            x1 = pos1[:,0:1]
            y1 = pos1[:,1:2]
            z1 = pos1[:,2:3]
            
            x1 -= np.mean(mass1 * x1) / np.mean(mass1)
            y1 -= np.mean(mass1 * y1) / np.mean(mass1)
            z1 -= np.mean(mass1 * z1) / np.mean(mass1)
            
            #polar coordinates (r, theta,z). We consider z the axis of rotation
            norm_r1=np.sqrt(x1**2 + y1**2)
            theta1=np.arctan(y1/x1)

            #total rotational velocity, tangential to radius. Let assume the axis of rotation is z
            v1rot=norm_r1*omega[0]
            v1rot_x=-v1rot*np.sin(theta1)
            v1rot_y=v1rot*np.cos(theta1)
            v1rot_z=np.zeros(N1) ### We are creating a zeros array b/c we assume that there is no velocity in the z direction since that is where we are rotating?
        
            ### I am pretty sure this is just a trick to use later... Have to investigate further
            v1rot_z=v1rot_z.reshape(N1,1)
        
            ### Have to investigate what hstack does again
            v1rot=np.hstack((v1rot_x,v1rot_y,v1rot_z))
    
            ### I believe this is randomizing the velocities of the number of
            ### particles you're looking at in a system (N, in this case). 
            vel1  =  np.random.randn(N1,3) 
        
            # in the frame of the centre of mass
            vel1 -= np.mean(mass1 * vel1,0) / np.mean(mass1)

            #vrot and random variation
            vel1=vel1+v1rot
            
            
            
            
            ### Now moving on to the second halo:
            
            x2 = pos2[:,0:1]
            y2 = pos2[:,1:2]
            z2 = pos2[:,2:3]
            
            x2 -= np.mean(mass2 * x2) / np.mean(mass2)
            y2 -= np.mean(mass2 * y2) / np.mean(mass2)
            z2 -= np.mean(mass2 * z2) / np.mean(mass2)
            
            #polar coordinates (r, theta,z). We consider z the axis of rotation
            norm_r2=np.sqrt(x2**2 + y2**2)
            theta2=np.arctan(y2/x2)

            #total rotational velocity, tangential to radius. Let assume the axis of rotation is z
            v2rot=norm_r2*omega[1]
            v2rot_x=-v2rot*np.sin(theta2)
            v2rot_y=v2rot*np.cos(theta2)
            v2rot_z=np.zeros(N2) ### We are creating a zeros array b/c we assume that there is no velocity in the z direction since that is where we are rotating?
        
            ### I am pretty sure this is just a trick to use later... Have to investigate further
            v2rot_z=v2rot_z.reshape(N2,1)
        
            ### Have to investigate what hstack does again
            v2rot=np.hstack((v2rot_x,v2rot_y,v2rot_z))
    
            ### I believe this is randomizing the velocities of the number of
            ### particles you're looking at in a system (N, in this case). 
            vel2  =  np.random.randn(N2,3) 
        
            # in the frame of the centre of mass
            vel2 -= np.mean(mass2 * vel2,0) / np.mean(mass2)

            #vrot and random variation
            vel2=vel2+v2rot
        
        else:
            vel1=np.zeros((N1,3)) 
            vel2=np.zeros((N2,3)) 
        
    
        return mass1, mass2, pos1, pos2, vel1, vel2
    
    else:   
        mass = mtot*np.ones((N,1))/N  # total mass of particles is mtot. all particles have the same mass here.
        pos  = np.random.randn(N,3)   # randomly selected positions from a normal distribution. 
        #Could be modified to take into account the initial density profile of the halo.
    
        if pure_radial_flow == True:
        
            # First, I need to find the distance (or magnitute) from the center:
            magnitude = np.sqrt(np.sum(pos**2, axis = 1, keepdims = True))
        
            # Now I have to find the radial velocities by multiplying the unit 
            # vector to the user-inputted radial velocity:
            vel_radial = radial_velocity * (pos / magnitude)
        
            # Including the Maxwell-Boltzmann distribution we calculated earlier.
            # We need to find normal random distributions since that is part of the
            # equation and will be used later to calculate momentum. I reference
            # the equation found on Wikipedia
            random_vel = mb_sigma * np.random.randn(N, 3)
        
            vel = vel_radial + random_vel
        
            # Now, take the mass weighted average velocities to ensure that the
            # analysis remains at the center of mass (the rest frame) by dividing 
            # by the total mass of the system. This also subtracts the center of
            # mass from every particle in the system. Net momentum should be 0 here
            vel -= np.mean(mass * vel, 0) / np.mean(mass)
        
        

        elif(v0==1):
            # for solid rotation: Vrot=radius*omega along e_theta. Let's calculate the radii of particles first
        
            x = pos[:,0:1]
            y = pos[:,1:2]
            z = pos[:,2:3]

            # in the frame of the centre of mass
            x -= np.mean(mass * x) / np.mean(mass)
            y -= np.mean(mass * y) / np.mean(mass)
            z -= np.mean(mass * z) / np.mean(mass)
        
            #polar coordinates (r, theta,z). We consider z the axis of rotation
            norm_r=np.sqrt(x**2 + y**2)
            theta=np.arctan(y/x)

            #total rotational velocity, tangential to radius. Let assume the axis of rotation is z
            vrot=norm_r*omega
            vrot_x=-vrot*np.sin(theta)
            vrot_y=vrot*np.cos(theta)
            vrot_z=np.zeros(N) ### We are creating a zeros array b/c we assume that there is no velocity in the z direction since that is where we are rotating?
        
            ### I am pretty sure this is just a trick to use later... Have to investigate further
            vrot_z=vrot_z.reshape(N,1)
        
            ### Have to investigate what hstack does again
            vrot=np.hstack((vrot_x,vrot_y,vrot_z))
    
            ### I believe this is randomizing the velocities of the number of
            ### particles you're looking at in a system (N, in this case). 
            vel  =  np.random.randn(N,3) 
        
            # in the frame of the centre of mass
            vel -= np.mean(mass * vel,0) / np.mean(mass)

            #vrot and random variation
            vel=vel+vrot
        # --------------
        else:
            #zero initial velocities in the frame of reference of the halo
            vel=np.zeros((N,3))    
       # --------------
    
    
        return mass, pos, vel


##----------------------------------------------------------------
##              Defining the accleration function:               -
##----------------------------------------------------------------


def getAcc(pos,
           mass,
           vel, 
           G, 
           softening, 
           t_start = None, 
           t_final = None, 
           include_expansion = False,
           comoving = False):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    NOTE: You can see that everything is put in matrix form, allowing for matrix operations rather than looping over particles 
    to get each update. This is not just because it looks cool to do all calculations in only one line rather than a FOR loop. 
    It  also significantly improves the computational performance in python!!
    """
    
    if include_expansion == True and t_start is not None:
        a, H_param_cu, distance = ScaleFactor(t_start, t_final, pos, add_expansion = True)
        
        if comoving == False:
            x = distance[:,0:1]
            y = distance[:,1:2]
            z = distance[:,2:3]
        else:
            
            x = pos[:, 0:1]
            y = pos[:, 1:2]
            z = pos[:, 2:3]
        
        
    else:    
        # positions r = [x,y,z] for all particles
        a = 1
        H_param_cu = 0.0
        x = pos[:,0:1]
        y = pos[:,1:2]
        z = pos[:,2:3]
    
    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations 
    """
    You can see that we included a "softening term". Its goal is to avoid getting an (near)infinite value when distance
    between two particles is ~ zero. It can happen in simulations where resolution, 
    number of particles and float precision is limited but would be unphysical. 
    We've seen that close-encounter collisions are rather irrelevant in collisionless systems likes haloes.
    So softening is essentially a user-defined resolution limit for numerical gravity. 
    """
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)
    
    # acceleration under gravity (Newton's second law) (notice we are calculating vec(r)/r^3 instead or 1/r^2 as we need the
    #direction of the force for each pair of particles)
    ### Need to look into what the @ does 
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass
    
    # pack together the acceleration components (hstack performs a concatenation)
    ### Ah this starts to answer my previous question
    accel_no_expansion = np.hstack((ax,ay,az))
    
    if include_expansion == True:
        acceleration = (accel_no_expansion / a**2) - 2 * H_param_cu * vel
        
    else:
        acceleration = accel_no_expansion
    return acceleration


##---------------------------------------------------------------
##                Defining the energy function:                 -
##---------------------------------------------------------------


def getEnergy( pos, vel, mass, G ):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    # Potential Energy:
    
    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]
    
    # matrix that stores all pairwise particle separations: r_j - r_i. Note that each pair appears twice: dx(i,j)=-dx(j,i)
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations 
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]
    
    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
    
    #Radial kinetic energy: first Convert to Center-of-Mass frame
    x -= np.mean(mass * x) / np.mean(mass)
    y -= np.mean(mass * y) / np.mean(mass)
    z -= np.mean(mass * z) / np.mean(mass)
    norm_r=np.sqrt(x**2 + y**2 + z**2)
    x1=x/norm_r
    y1=y/norm_r
    z1=z/norm_r
    r=np.hstack((x1,y1,z1))
    vel_r=np.sum(vel*r,axis=1)
    vr2=vel_r**2
    N=vr2.shape[0]
    vr2=np.reshape(vr2,(N,1))
    KE_rad = 0.5 * np.sum(mass * vr2,axis=0 )
    KE_rad=KE_rad.reshape(1)[0]
    
    #Orbital kinetic energy
    KE_orb=KE-KE_rad
    
    return KE, PE, KE_rad, KE_orb


##-------------------------------------------------------------------------------
##  Defining the main function. This is the 'heavy lifter' of the function by   -
##  calling all the previous functions and calculating the simulation           -
##-------------------------------------------------------------------------------


def main(N_input, 
         plotRealTime_input = True, 
         omega_input = 0.1,
         tEnd_input = 10.0,
         mtot_input = 20,
         softening_input = 0.1,
         add_expansion_input = False,
         comoving_input = False,
         rad_flow_input = False,
         initial_radial_velocity = 0,
         temp_input = 1):
    """ N-body simulation """
    
    # Setting up the time it takes 
    # Simulation parameters ----------------------------------------------------
    N         = N_input    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = tEnd_input   # time at which simulation ends
    dt        = 0.01   # timestep
    softening = softening_input    # softening length
    G         = 1.0    # Newton's Gravitational Constant. Here set to 1 in code units for covenience.
    mtot      =  mtot_input # Total mass of the object
    plotRealTime = plotRealTime_input # switch on for plotting as the simulation goes along
    omega        = omega_input  # initial angular velocity if solid rotation
    v0           = 1.0   #if not 1.0, set initial velocitites to 0
    add_expansion = add_expansion_input # Adds in expansion factor or does sim without it
    add_comoving  = comoving_input # If expansion is added, calculates expansion in comoving or physical distances
    rad_flow  = rad_flow_input # If you want to include radial flow, set this parameter equal to True
    init_rad_flow = initial_radial_velocity # Input for the initial radial velocity of the system
    temperature = temp_input # Adding in the temperature to calculate Maxwell Boltzmann distributions
    #-------------------------------------------------------------------------------
    
    
    # Set where to store outputs-------------------------------------------------------------------------------    
    # Create a directory for the Simulation
    directory = "Testing N-Body Sim/Image Folder"
    # Parent Directory path: change to your own path
    parent_dir = "/Users/RachelCampo/Desktop/CUNY Classes/Fall 2025 Galaxies/fall-2025-galaxies-homework/HW4-instructions"
    isdir = os.path.isdir(parent_dir)
    if isdir==False: # create parent directory if it does not exist.
        os.mkdir(parent_dir)
    # Make complete Path
    path = os.path.join(parent_dir, directory)
    #Create directory
    isdir = os.path.isdir(path)
    if isdir==False:
        os.mkdir(path)
    #-----------------------------------------------------------------------------

    if len(N) > 1:
        N_total = sum(N)
    else:
        N_total = N

    #Now let's run the simulation!
    ic_results = InitialConditions(N,
                                   omega,
                                   mtot,
                                   v0,
                                   radial_velocity = init_rad_flow,
                                   pure_radial_flow = rad_flow,
                                   temp = temperature) #load initial conditions
    
    if len(ic_results) == 6:
        
        mass1, mass2, pos1, pos2, vel1, vel2 = ic_results
        
        mass = np.vstack((mass1, mass2))
        pos = np.vstack((pos1, pos2))
        vel = np.vstack((vel1, vel2))
    
    # calculate initial gravitational accelerations
    acc = getAcc(pos, 
                 mass, 
                 vel, 
                 G, 
                 softening, 
                 comoving = add_comoving,
                 include_expansion = add_expansion,
                 t_start = t,
                 t_final = tEnd)
    
    # calculate initial energy of system
    KE, PE, KE_rad, KE_orb  = getEnergy( pos, vel, mass, G )
    
    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    
    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N_total,3,Nt+1))
    pos_save[:,:,0] = pos
    KE_save = np.zeros(Nt+1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt+1)
    PE_save[0] = PE
    KE_Rsave = np.zeros(Nt+1)
    KE_Rsave[0] = KE_rad
    KE_Osave = np.zeros(Nt+1)
    KE_Osave[0] = KE_orb
    t_all = np.arange(Nt+1)*dt
    
    ### RLC EDIT ###
    # I'm going to add in an array that tracks when the virialization energy is
    # equal to 0:
    virialized_save = np.zeros(Nt + 1)
        
    
    # prep figure
    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])

    # Simulation Main Loop
    """
    We are using a numerical (=approximate) scheme to solve the equation of motion.
    The basic idea is that for each discrete timestep we calculate the acceleration at start time t and consider it constant
    during the timestep duration Dt. So we can calculate the velocity and displacement of the particle easily. 
    With the simplest version of this (1 coarse timestep= 1 acceleration update, then 1 velocity update, then 1 "drift"
    (displacement) update across Dt), errors tend to pile up over timesteps and you end up far from the solution. 
    A more stable scheme is used here, the kick-drift-kick version of the Leapfrog scheme. Can you explain how it is different?
    """
    for i in range(Nt):
            
        # (1/2) kick
        vel += acc * dt/2.0
        
        # drift
        pos += vel * dt
        
        # update accelerations
        acc = getAcc( pos, mass, vel, G, softening )
        
        # (1/2) kick
        vel += acc * dt/2.0
        
        # update time
        t += dt
        
        # get energy of system
        KE, PE, KE_rad, KE_orb  = getEnergy( pos, vel, mass, G )
        
        # Checking to see if the system has virialized:
        if 2 * KE + PE == 0:
            virialized_save[i] = 1
        
        # save energies, positions for plotting trail
        pos_save[:,:,i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE
        KE_Rsave[i+1] = KE_rad
        KE_Osave[i+1] = KE_orb
        
        # plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:,0,max(i-50,0):i+1]
            yy = pos_save[:,1,max(i-50,0):i+1]
            plt.scatter(xx,yy,s=1,color=[.7,.7,1])
            plt.scatter(pos[:,0],pos[:,1],s=10,color='red')
            ax1.set(xlim=(-5, 5), ylim=(-5, 5))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
            ax1.set_yticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
            
            plt.sca(ax2)
            plt.cla()
            ax2.scatter(t_all,KE_save,color='red',s=1,label='KE')
            ax2.scatter(t_all,PE_save,color='blue',s=1,label='PE')
            ax2.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot')
            ax2.scatter(t_all,2*KE_save+PE_save,color='pink',s=1,label='2KE+PE' )   
            ax2.scatter(t_all,KE_Rsave,color='green',s=1,label='KE_rad')
            ax2.scatter(t_all,KE_Osave,color='orange',s=1,label='KE_orb')
            ax2.legend(bbox_to_anchor=(1.35, 1.35), loc='upper right', borderaxespad=0)
            ax2.set(xlim=(0, tEnd+3), ylim=(-300, 300))
            ax2.set_aspect(0.005)
            
            # add labels/legend
            plt.sca(ax2)
            plt.xlabel('time')
            plt.ylabel('energy')
            
            plt.sca(ax1)
            plt.xlabel('x')
            plt.ylabel('y')
           
            
            #What can you say about the validity of the virial theorem vs that of the conservation of mechanical energy?
            # Save figure
            plt.savefig(str(path) +'/nbody_' + str(N) + 'part_omega' + str(omega) + '_step' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
   
            plt.pause(0.001)
            
    plt.show()
    

    
    return virialized_save


##---------------------------------------------------------------
##              Defining the movie making function:             -
##---------------------------------------------------------------


def make_movie(omega, N, extra = None):

   # Create a directory for the Simulation
   directory = "Testing N-Body Sim/Image Folder"
   # Parent Directory path: change to your own path
   parent_dir = "/Users/RachelCampo/Desktop/CUNY Classes/Fall 2025 Galaxies/fall-2025-galaxies-homework/HW4-instructions"
   # directory = "MyNbodyRun_Om"+str(omega)+"_N"+str(N)+"part"
    # Parent Directory path: change to your own path
    #parent_dir = "/Users/charlottewelker/Desktop/NbodyMS/"
   image_folder = os.path.join(parent_dir, directory)
   
   image_name_pattern = "*" + str(N) + "part*omega" + str(omega) + "*.png"
   image_files = glob.glob(os.path.join(image_folder, image_name_pattern))
    
   fps=10 #number of frames per second
#    image_files = [os.path.join(image_folder,img)
#                    for img in os.listdir(image_folder)
#                    if img.endswith(".png")]
   image_files_sorted = natsorted(image_files,reverse=False)
   clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files_sorted, fps=fps)
   clip.write_videofile(str(parent_dir) + "/Testing N-Body Sim/Movie Folder/" + str(extra) + '-nbody-cluster_Om'+str(omega)+'_N'+str(N)+'part.mov')
   return 0


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

####################################################################################
####################################################################################
###                                                                              ###
###  PART A) HOW DOES THE COMPUTATIONAL TIME VARY WITH THE NUMBER OF PARTICLES?  ###
###                                                                              ###
####################################################################################
####################################################################################


###  I'm going to compare the run times of different N particles by choosing N   
###  to be 20, 60, 80, 100, and 120                                              


##---------------------------------------------------------------
##          Starting with 20 particles in the system:           -
##---------------------------------------------------------------

#-------------------------------
start_time = time.time()

if __name__== "__main__":
  test20 = main(N_input = 20)
  
end_time = time.time()

total_time = end_time - start_time
#-------------------------------

print(total_time)
print(test20)
  
make_movie(0.1, 20)

##----------------------------------------------------------------
##  Using N = 20 took approximately 2 minutes and 57 seconds     -
##----------------------------------------------------------------


##----------------------------------------------------------------
##          Now moving onto 60 particlces in the system:         -
##----------------------------------------------------------------

#-------------------------------
start_time = time.time()

if __name__ == "__main__":
  test60 = main(N_input = 60)

end_time = time.time()

total_time = end_time - start_time
#-------------------------------

print(total_time)
print(test60)
  
make_movie(0.1, 60)


##---------------------------------------------------------------
##    Using N = 60 took approximately 3 minutes and 3 seconds   -
##---------------------------------------------------------------


##---------------------------------------------------------------
##          Now looking at 80 particles in the system           -
##---------------------------------------------------------------

#-------------------------------
start_time = time.time()

if __name__ == "__main__":
  test80 = main(N_input = 80)

end_time = time.time()

total_time = end_time - start_time
#-------------------------------

print(total_time)
print(test80)
  
make_movie(0.1, 80)

##---------------------------------------------------------------
##    Using N = 80 took approximately 3 minutes and 4 seconds   -
##---------------------------------------------------------------


##----------------------------------------------------------------
##            Now using 100 particles in the system:             -
##----------------------------------------------------------------

#-------------------------------
start_time = time.time()

if __name__ == "__main__":
 test100 = main(N_input = 100)
  
end_time = time.time()

total_time = end_time - start_time
#-------------------------------

print(total_time)
print(test100)
  
make_movie(0.1, 100)

##---------------------------------------------------------------------
##  Using N = 100 took appriximately 3 minutes and 8 secodns to run   -
##---------------------------------------------------------------------


##----------------------------------------------------------------
##            Now testing 120 particles in the system:           -
##----------------------------------------------------------------

#-------------------------------
start_time = time.time()

if __name__ == "__main__":
  test120 = main(N_input = 120)
  
end_time = time.time()

total_time = end_time - start_time
#-------------------------------

print(total_time)
print(test120)
  
make_movie(0.1, 120)


###################################################################################
###################################################################################
###                                                                             ###
###  PART A) HOW DOES IT CHANGE ITS EVOLUTION? HOW MANY TIME STEPS DO YOU NEED  ###
###                        TO REACH VIRIAL EQUILIBRIUM?                         ###
###                                                                             ###
###################################################################################
###################################################################################


###  Now I am going to compare how changing the solid angle affects the system   
###  and determine how many time steps I need in order to reach virialization.   


##----------------------------------------------------------------
##              Starting with N = 20 and Omega = 1               -
##----------------------------------------------------------------

if __name__ == "__main__":
    virialize_test_1 = main(N_input = 20,
                            plotRealTime_input = True,
                            omega_input = 1)
    
print(virialize_test_1)

make_movie(1, 20)


###  I noticed that at higher omega values, the system virializes faster. I   
###  will try a few more omega values to confirm.   

###  It also looks like the lower omega values need longer time steps to          
###  virialize. It looks like an omega of 1 virializes quickly and stays          
###  virialized for essentially the rest of the time steps, but I'm not           
###  confident that an omega with 0.1 stays virialized, it goes too far off of    
###  the horizontal zero line later in the sim for me to consider it virialized      

 
##---------------------------------------------------------------
##                Now using N = 20 and Omega = 10               -
##---------------------------------------------------------------

if __name__ == "__main__":
    virialize_test_10 = main(N_input = 20,
                            plotRealTime_input = True,
                            omega_input = 10)
    
print(virialize_test_10)

make_movie(10, 20)


###  Setting too high of omega (with this amount of particles) seems to never   
###  virialize. Probably depending on the number of particles, omega needs to   
###  be in a 'goldilocks' region                                                

##---------------------------------------------------------------
##      Using N = 20 and Omega = 0.1 and an End Time = 40       -
##---------------------------------------------------------------

if __name__ == "__main__":
    virialize_test_10_40 = main(N_input = 20,
                            plotRealTime_input = True,
                            omega_input = 0.1,
                            tEnd_input = 40)
    
print(virialize_test_10_40)

make_movie(0.1, 20)


##------------------------------------------------------------------------------
##  It looks like when Omega = 0.1, the system virializes when time = 5, but   -
##  then begins to move away from virialization after that. With omega = 1,    -
##  the system appears to virialize at around time = 2 and mostly stays        -
##  virialized after                                                           -
##------------------------------------------------------------------------------


###########################################################################
###########################################################################
###                                                                     ###
###           PART A: MODIFY THE TOTAL MASS TWICE AND COMPARE           ###
###                                                                     ###
###########################################################################
###########################################################################


###  I will continue to use N = 20 with Omega = 0.1, however I will use total   
###  mass of the system to be 10 and 100                                        


##---------------------------------------------------------------
##                Starting with total mass = 10                 -
##---------------------------------------------------------------


if __name__ == "__main__":
    mass_10 = main(N_input = 20,
                   plotRealTime_input = True,
                   omega_input = 0.1,
                   tEnd_input = 10,
                   mtot_input = 10)

make_movie(0.1, 20)


###  With this mass and initial conditions, the system virializes almost      
###  instantly and remains virialized for the entire time of the simulation   

##----------------------------------------------------------------
##                Now examining total mass = 100                 -
##----------------------------------------------------------------

if __name__ == "__main__":
    mass_10 = main(N_input = 20,
                   plotRealTime_input = True,
                   omega_input = 0.1,
                   tEnd_input = 10,
                   mtot_input = 100)

make_movie(0.1, 20)


###  With higher masses and these initial conditions, the system never   
###  virializes and the particles fling outward into space               


###########################################################################
###########################################################################
###                                                                     ###
###        PART A: MODIFY THE SOFTENING LENGTH TWICE AND COMPARE        ###
###                                                                     ###
###########################################################################
###########################################################################


##---------------------------------------------------------------
##            Starting with a softening parameter = 1           -
##---------------------------------------------------------------

if __name__ == "__main__":
    mass_10 = main(N_input = 20,
                   plotRealTime_input = True,
                   omega_input = 0.1,
                   tEnd_input = 10,
                   mtot_input = 100,
                   softening_input = 1)
    
make_movie(0.1, 20)
    

##---------------------------------------------------------------
##          Now looking at a softening parameter = 10           -
##---------------------------------------------------------------

if __name__ == "__main__":
    mass_10 = main(N_input = 20,
                   plotRealTime_input = True,
                   omega_input = 0.1,
                   tEnd_input = 10,
                   mtot_input = 100,
                   softening_input = 10)
    
make_movie(0.1, 20)


###  It appears that lower softening lengths become more virialized than higher   
###  softening lengths. I think this is because if you put too high of a          
###  softening length, then it fundamentally changes where the sim puts the       
###  particles. For example, if you have 1 / 10 + 0.01, the calculated number     
###  is approximately 0.1, however if your softening length instead is 1/ 10 +    
###  5, that's a higher softening length, you'll get approximately 0.06, which    
###  makes the softening parameter more noticable, and thus making significant    
###  modifications to the calculations.                                           


############################################################################
############################################################################
###                                                                      ###
###         ADD THE EXPANSION OF THE UNIVERSE WHILE USING N = 50         ###
###                                                                      ###
############################################################################
############################################################################


##---------------------------------------------------------------
##          Adding In Expansion In Physical Distances           -
##---------------------------------------------------------------

if __name__ == "__main__":
    add_expansion = main(N_input = 50,
                         plotRealTime_input = True,
                         omega_input = 0.1,
                         tEnd_input = 10,
                         mtot_input = 100,
                         softening_input = 0.01,
                         add_expansion_input = True)
    
make_movie(0.1, 50, "Testing Physical Distance Expansion")


##----------------------------------------------------------------
##          Looking at Expansion in Comoving Distances           -
##----------------------------------------------------------------

if __name__ == "__main__":
    add_expansion = main(N_input = 50,
                         plotRealTime_input = True,
                         omega_input = 0.1,
                         tEnd_input = 10,
                         mtot_input = 100,
                         softening_input = 0.01,
                         add_expansion_input = True,
                         comoving_input = True)
    
make_movie(0.1, 50, "Testing Comoving Expansion")


###  When looking at physical vs. comoving when expansion is involved, because    
###  I chose to look at redshifts between 2 - 3, you can certainly see the        
###  galaxies 'running away' from each other. Because we are looking over         
###  proper distances, you can see the expansion of the Universe. All particles   
###  are knocked out of view when time is approximately 2, however in the         
###  comoving point of view, expansion is still present because you do see some   
###  galaxies being ejected outward and off the graph, however, some galaxies     
###  still remained clumped together, displaying how the gravitational            
###  potential is winning over the acceleration of expansion.                     


###########################################################################
###########################################################################
###                                                                     ###
###                    MODIFY THE INITIAL CONDITIONS                    ###
###                                                                     ###
###########################################################################
###########################################################################


##-------------------------------------------------------------------------------
##  Testing Pure Radial Flow with Omega = 0 and Radial Flow = 0.5 in Comoving   -
##  Coordinates                                                                 -
##-------------------------------------------------------------------------------


if __name__ == "__main__":
    add_expansion = main(N_input = 50,
                         plotRealTime_input = True,
                         omega_input = 0,
                         tEnd_input = 10,
                         mtot_input = 100,
                         softening_input = 0.01,
                         add_expansion_input = True,
                         comoving_input = True,
                         rad_flow_input = True,
                         initial_radial_velocity = 0.5)
    
make_movie(0, 50, "Testing Pure Radial Flow with Expansion in Comoving")

##---------------------------------------------------------------
##      Now Testing the Radial Flow in Physical Distances       -
##---------------------------------------------------------------

if __name__ == "__main__":
    add_expansion = main(N_input = 50,
                         plotRealTime_input = True,
                         omega_input = 0,
                         tEnd_input = 10,
                         mtot_input = 100,
                         softening_input = 0.01,
                         add_expansion_input = True,
                         comoving_input = False,
                         rad_flow_input = True,
                         initial_radial_velocity = 0.5)
    
make_movie(0, 50, "Testing Pure Radial Flow with Expansion in Physical Distances")


###  It looks like when adding in radial flow, the particles seem to not fly   
###  away from each other as easily, especially in the beginning of the        
###  simulation, although with my particular initial conditions, it doesn't    
###  look like the system reaches virialization within the timeframe           


##--------------------------------------------------------------------
##  Now Adding in a Maxwell Boltzman Distribution for the Velocity   -
##  Distribution in Comoving:                                        -
##--------------------------------------------------------------------

if __name__ == "__main__":
    add_expansion = main(N_input = 50,
                         plotRealTime_input = True,
                         omega_input = 0,
                         tEnd_input = 10,
                         mtot_input = 100,
                         softening_input = 0.01,
                         add_expansion_input = True,
                         comoving_input = True,
                         rad_flow_input = True,
                         initial_radial_velocity = 0.5)
    
make_movie(0, 50, "Testing Maxwell Boltzmann Distribution in Comoving")


##-------------------------------------------------------------------------------
##  Now Adjusting the Initial Conditions so there are Two Peaks with  N = 100   -
##-------------------------------------------------------------------------------

if __name__ == "__main__":
    add_expansion = main(N_input = [50, 50],
                         plotRealTime_input = True,
                         omega_input = [0, 1],
                         tEnd_input = 10,
                         mtot_input = [50, 50],
                         softening_input = 0.01,
                         add_expansion_input = True,
                         comoving_input = True,
                         rad_flow_input = True,
                         initial_radial_velocity = [0.5, 0.5])
    
make_movie([0,1], [50,50], "Testing Two Halo Peaks in Comoving")


##-------------------------------------------------------------------------
##  Changing the Angular and Radial Velocities to Determine Differences   -
##-------------------------------------------------------------------------

if __name__ == "__main__":
    add_expansion = main(N_input = [50, 50],
                         plotRealTime_input = True,
                         omega_input = [1, 1],
                         tEnd_input = 10,
                         mtot_input = [50, 50],
                         softening_input = 0.01,
                         add_expansion_input = True,
                         comoving_input = True,
                         rad_flow_input = True,
                         initial_radial_velocity = [0.1, 0.1])
    
make_movie([1,1], [50,50], "Testing Two Halo Peaks in Comoving")


###  It looks like when I increased the angular velocity and decreased the        
###  radial, less particles were ejected from the system and more of the merged   
###  particles stayed together relatively well, however I'm sure if I increased   
###  the time steps, I may see that the particles that got 'flung out' may        
###  adjust their trajectories because of the stronger gravitational pull by      
###  the particles that stayed together. However, I noticed on both runs that     
###  neither system virialized. That may be due to the fact that I didn't         
###  increase the step size.                                                      

