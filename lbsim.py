
# This code uses the lattice-boltzmann distrivution (https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods) to simulate the flow of some kind of liquid in a tunnel.
# So far it is capable of plotting the rotation, velocities, density distribution with a stationary or moving barrier
# as long as the barrier moves significantly slower than the liquid's velocities (i.e it is handled as a stationary
#object that is in a different position in every frame, and the movement doesn't accelerate or "push" the fluid around.)


# FUTURE PLANS: method for a picture to change the barrier, initial changes in density and velocity as well as later an interactive panel for these.
# I also plan to implement the whole thing in 3d (based onhttps://www.sciencedirect.com/science/article/pii/S0898122107006256?via%3Dihub),
#though this would need a bit more thought.
# Later more barriers and boundaries should also be considered.

# I know this method is crudely simplified - as in it uses statistics to work around solving the EOS, but as
# a fun little project I consider it sufficient.


# Sidenote: I'm thinking of rewriting the whole ordeal in some more object oriented way
#(the "whole fluid" being the object and all the different stuff methods for it... shouldn't be that hard)
# but at first I want to make sure that it works the way it is
import numpy as np
from time import process_time
from matplotlib import pyplot as plt
from matplotlib import animation as animation
import math


chamber_height = 80					#these are the dimensions of the chamber
chamber_width = 200
viscosity = .02				# fluid viscosity - this can be varied through input later
relax_param = 1 / (3*viscosity + .5)		# The relaxation parameter - in essence this controls the timescale on which the values converge to equilibrium. It comes from the viscosity
v0_horiz = 0.1				# horizontal initial speed (and the speed from "left"). Vertical is not set, but it is easy to include. doesn't make a difference though
do_i_want_rotation=False
need_curl=True # for now - this is a dummy, I can easily write a function to change it from either command line arg or just as an input
file_name="turbulence_curl"

#  density - it starts uniformly, but it is possible to change this - I'Ll add a function later that can do so - since there is no "unit" of density, it can be any number - might as well have it be 1.
velocities=np.ones((9,chamber_height,chamber_width)) #these are the velocities in a big np array. velocities[0] is the density  that stays, [1:8] are the vectors in the order: "up", "right", "down", "left", up-right, down-right, down-left, up-left.
####### Soooooo the discretized distribution of velocities should look like: 1+3(e_i*u)+9/2(e_i*u)^2-(3/2) u^2  - with e_i being the vector along the "step", u is the horizontal velocity
# the first two terms should be meaningless when :
# a) the velocity is 0 (velocity[0]), or perpendicular (velocity[1 and 3]) to u - which is the flow velocity.
#after this in mind, let's set the L-B velocities

#estabilishing particle densities  along the directions
#where only the last component has effect
# food for thought: I probably should've initiated velocities as signed values (positive - negative) or better yet, complex numbers but I guess I'll leave them as they are. Especially since for 3D cplx wouldn't make sense anymore.
#  I could still use signed values tho...
velocities[0]=4/9*(velocities[0]-(3/2)*v0_horiz**2)
#velocities[0]
velocities[1]=1/9*(velocities[1]-(3/2)*v0_horiz**2)
velocities[3]=1/9*(velocities[3]-(3/2)*v0_horiz**2)
# where all three comps have effect, 1/9 distribution
velocities[2]=1/9*(velocities[2]-(3/2)*v0_horiz**2+(9/2)*v0_horiz**2+3*v0_horiz)
velocities[4]=1/9*(velocities[4]-(3/2)*v0_horiz**2+(9/2)*v0_horiz**2-3*v0_horiz) #minus sign is since the two vectors poiint towards opposite directions
# 1/36th distribution
sign=[1,1,-1,-1]#I probably should give this a "more abstract" name - so that I could call something more trivial "sign" later...
for direction in range(5,9):
	velocities[direction]=(1/36)*(velocities[direction]-(3/2)*v0_horiz**2+(9/2)*v0_horiz**2+ 3*v0_horiz*sign[direction-5])#this last part is a rather complicated way of adding a +/- sign based on direction, but I think it will do

rho =np.zeros(velocities[0].shape) # I'll return this at the end - I will no longer need the old rho
for pseu in range(9):
	rho+=velocities[pseu]

sum_vel_x= (velocities[2] + velocities[5] + velocities[6] - velocities[4] - velocities[7] - velocities[8]) / rho # these expressions will need something better once we reach 3D or they'll grow to be HUGE
sum_vel_y= (velocities[1] + velocities[5] + velocities[8] - velocities[3] - velocities[6] - velocities[7]) / rho #



obstacles=np.zeros((9,chamber_height, chamber_width), bool) #okay, so this should give the obstacles. True means that something is on that 'pixel'. Now, I have 9 of this because... well, I'll use later the "above" "under" etc cells
# with the collision, and I think it is faster if I calculate the whole bunch only once. Numbering - conveniently - mean the same directions as the velocities.

#let's add a wall! - ummm... no, not like that!
def add_wall(starting_coord, ending_coord, obstacles):# so I hope for this to later be more complicated, but right now it can add a "straight" line of wall between two points to the obstacle or remove it and... kind of that's it. Should only work in 2D
	obstacles[0][starting_coord[0]][starting_coord[1]]=True
	obstacles[0][ending_coord[0]][ending_coord[1]]=True
	wall_ylen=abs(starting_coord[0]-ending_coord[0])
	wall_xlen=abs(starting_coord[1]-ending_coord[1])
	if wall_xlen>=wall_ylen:# i need to go with the longer "side"
		for i in range(1,wall_xlen):# start from 1, since the starting point is already "drawn"
			xcoor=starting_coord[1]+i
			ycoor=starting_coord[0]+abs(i/wall_xlen*wall_ylen)
			obstacles[0][int(ycoor)][int(xcoor)]=True
	else:
		for i in range(1,wall_ylen):# start from 1, since the starting point is already "drawn"
			xcoor=starting_coord[1]+abs(i/wall_ylen*wall_xlen)
			ycoor=starting_coord[0]+i
			obstacles[0][int(ycoor)][int(xcoor)]=True

add_wall([10,40],[48,40], obstacles)# this is for the turbulence - I could add a more complicated one but this'll do.
obstacles

#now comes the moving of my particles - it should be easy, every single direction should just move one - for the diagonals ofc this means two "roll"-s
def move_once( velocities, obstacles): #these are the things that directly participate in moving the particles. I think everything else has an indirect effect through the effect on velocity.
# Later if the obstacles move and "push" the matter, that will be another argument
	for pseu in [1,8,5]:#upwards rolls
		velocities[pseu]=np.roll(velocities[pseu],1, axis=0)
	for pseu in [3,6,7]:#downward rolls
		velocities[pseu]=np.roll(velocities[pseu],-1, axis=0)
	for pseu in [5,2,6]:#right rolls
		velocities[pseu]=np.roll(velocities[pseu],1, axis=1)
	for pseu in [8,4,7]:#left rolls
		velocities[pseu]=np.roll(velocities[pseu],-1, axis=1)
	#But I also need to move around the obstacles! for now, obstacles can only perform "perfect collisions", meaning the velocity is inverted here.
	#Basically what this means is that there is always material "inside" the barrier (otherwise the division with density would be a very bad idea), but because of this part,
	# it kind of cannot "escape" i.e. after e.g. the "upwards" moving particles leave upwards, they get replaced with the particles that travelled "downward" in that very step.
	#Yes, it would be neater to not even let those particles move (or not calculate them in the first place) - but I use np methods because they are fast and easy - i'd have to write
	#custom functions which are at best time consuming to write and at worst won't even be as efficient as numpy.
	velocities[3][obstacles[3]] = velocities[1][obstacles[0]]
	velocities[4][obstacles[4]] = velocities[2][obstacles[0]]
	velocities[1][obstacles[1]] = velocities[3][obstacles[0]]
	velocities[2][obstacles[2]] = velocities[4][obstacles[0]]
	velocities[7][obstacles[7]] = velocities[5][obstacles[0]]
	velocities[8][obstacles[8]] = velocities[6][obstacles[0]]
	velocities[5][obstacles[5]] = velocities[7][obstacles[0]]
	velocities[6][obstacles[6]] = velocities[8][obstacles[0]]




def around_obstacles(obstacles):# this needs to be called right before the simulation, after all needed obstacles are added. yes, I copied this whole thing from the above function. I define the obstacle "borders" used in the above func.
	for pseu in [1,8,5]:#upwards rolls
		obstacles[pseu]=np.roll(obstacles[0],1, axis=0)
	for pseu in [3,6,7]:#downward rolls
		obstacles[pseu]=np.roll(obstacles[0],-1, axis=0)
	for pseu in [5,2,6]:#right rolls
		obstacles[pseu]=np.roll(obstacles[0],1, axis=1)
	for pseu in [8,4,7]:#left rolls
		obstacles[pseu]=np.roll(obstacles[0],-1, axis=1)


starting_point=[int(chamber_height/2),int (chamber_width/2)]
dgree=0
barr_len=8
midp=[chamber_height/2, chamber_width/2]
def rotate_barrier(rot, length_barr, midp, obstacle):# this... well, introduces a single wall and rotates it. later barriers could be their own objects and startingpoints to rotate around.
	obstacle[0] = np.zeros((chamber_height,chamber_width), bool)					# True wherever there's a barrier
	if rot ==0:
		obstacle[0][(int(chamber_height/2)):int((chamber_height/2)+len), int(chamber_width/2)] = True
	else:
		obstacle[0][int(midp[0]), int(midp[1])]=True
		for cell in range(int(length_barr)):
			idxa=round((cell+1)*math.cos(rot))
			idxb=round((cell+1)*math.sin(rot))
			obstacle[0][int(idxa+chamber_height/2), int(idxb+chamber_width/2)]=True
			around_obstacles(obstacle)




def collide( vel, relax_param, v0_horiz):#now redistribute among the vectors
	new_rho =np.zeros(vel[0].shape) # I'll return this at the end - I will no longer need the old rho
	for pseu in range(9):
		new_rho+=vel[pseu]
	sum_vel_x= (vel[2] + vel[5] + vel[6] - vel[4] - vel[7] - vel[8]) / new_rho #these are also returnables
	sum_vel_y= (vel[1] + vel[5] + vel[8] - vel[3] - vel[6] - vel[7]) / new_rho #


	uxsq = sum_vel_x * sum_vel_x				# I'll use these squared etc. values a bunch of times, this is faster
	uysq = sum_vel_y * sum_vel_y				# I doubt that I found the most efficient way for all of these - it could be further optimized
	svx3= 3*sum_vel_x
	svy3= 3*sum_vel_y
	usq_sum = uxsq + uysq
	vel_mult = sum_vel_x * sum_vel_y


	vel[0] = relax_param * ( 4/9*( 1 - 1.5*usq_sum))*new_rho + (1-relax_param)*vel[0]
	vel[1] = relax_param * ( 1/9*(1 - 1.5*usq_sum + 4.5*uysq + svy3 ))*new_rho + (1-relax_param)*vel[1]
	vel[2] = relax_param * ( 1/9*(1 - 1.5*usq_sum + 4.5*uxsq + svx3 ))*new_rho + (1-relax_param)*vel[2]
	vel[3] = relax_param * ( 1/9*(1 - 1.5*usq_sum + 4.5*uysq - svy3 ))*new_rho + (1-relax_param)*vel[3]
	vel[4] = relax_param * ( 1/9*(1 - 1.5*usq_sum + 4.5*uxsq - svx3 ))*new_rho + (1-relax_param)*vel[4]
	vel[5] = relax_param * ( 1/36*(1 - 1.5*usq_sum + 4.5*(usq_sum + 2*vel_mult ) + svx3 + svy3  ))*new_rho + (1-relax_param)*vel[5]
	vel[6] = relax_param * ( 1/36*(1 - 1.5*usq_sum + 4.5*(usq_sum - 2*vel_mult ) + svx3 - svy3  ))*new_rho + (1-relax_param)*vel[6]
	vel[7] = relax_param * ( 1/36*(1 - 1.5*usq_sum + 4.5*(usq_sum + 2*vel_mult ) - svx3 - svy3  ))*new_rho + (1-relax_param)*vel[7]
	vel[8] = relax_param * ( 1/36*(1 - 1.5*usq_sum + 4.5*(usq_sum - 2*vel_mult ) - svx3 + svy3  ))*new_rho + (1-relax_param)*vel[8]
	# and now at the borders I just copy the initial setup since... well, I want them to be kept around the edges - I don't need the np.ones thingie, since the structure exists
#	velocities[1]=velocities[1]*1/9-(3/2)*v0_horiz**2
#	velocities[3]=velocities[3]*1/9-(3/2)*v0_horiz**2
#these are actually not needed since I have no "steady flow" vertically sooo...
	# where all three comps have effect, 1/9 distribution
	velocities[2][:,0]=1/9*(1-(3/2)*v0_horiz**2+(9/2)*v0_horiz**2+3*v0_horiz)
	velocities[4][:,0]=1/9*(1-(3/2)*v0_horiz**2+(9/2)*v0_horiz**2-3*v0_horiz) #minus sign is since the two vectors poiint towards opposite directions
	# 1/36th distribution
	sign=[1,1,-1,-1]#I probably should give this a "more abstract" name - so that I could call something more trivial "sign" later...
	for direction in range(5,9):
		velocities[direction][:,0]=(1/36)*(1-(3/2)*v0_horiz**2+(9/2)*v0_horiz**2+ 3*v0_horiz*sign[direction-5])#this last part is a rather complicated way of adding a +/- sign based on direction, but I think it will do
	#finally, return rho and the vel x,y
	return [new_rho, sum_vel_x, sum_vel_y]




#yeeee, so far I'm done with the basic functions, let's see if it works a.k.a: PLOTTING
this_figure = plt.figure(figsize=(8,3)) #24-9 scaled down
if need_curl:
	def plot_func(vx, vy, rho):#calculate curl - for plotting purposes
		crl=np.roll(vx,1,axis=0) -  np.roll(vx,-1,axis=0) +  np.roll(vy,-1,axis=1) - np.roll(vy,1,axis=1)
		return crl
	norm=[-.1, .1]
else:
	def plot_func(vx, vy, rho):
		return rho
	norm=[0.85, 1.15] # this is about the realistic range for rho I guess

#oh god how much time I spent trying to make work terrible alternatives before imshow....
the_fluid = plt.imshow(plot_func(sum_vel_x, sum_vel_y, rho), origin='lower', norm=plt.Normalize(norm[0], norm[1]),
									cmap=plt.get_cmap('jet'), interpolation='spline36') #jet or gist_rainbow are my favourites - let's use jet for now
#as for interpolation: I probably don't need it - and should I have performance issues i'll remove it.
ostacle_image = np.zeros((chamber_height, chamber_width,4), np.uint8)	# uint - 0-255 since rgba images... I have no idea why I thought I'd need 9, but now it is correct
ostacle_image[obstacles[0],3] = 255
image_obs = plt.imshow(ostacle_image, origin='lower', interpolation='none')# I don't want to have interpolation here
around_obstacles(obstacles)
#oookay so now comes an ugly part which I probably should rephrase. basically I'll define the "Iter_func" which iterates
# but I have two versions of it (so far) - one that rotates the barrier and one that does not - but that results in turbulence which is kind of more interesting

if do_i_want_rotation:
	dgree=0
	def iter_func(arg):							# arg is probably the frame number - I on't need it, but this is how animate works - oh well
		global dgree # I'll use globals here, I don't want to mess around with lambda... maybe this can be patched later?
		dgree+=.01
		global obstacles
		global image_obs, the_fluid, this_figure # the fig and the fluid prolly don't need to be "passed" here
		global  rho, sum_vel_x, sum_vel_y
		image_obs.remove()# remove the previous position of the obstacle - one could argue that this should come after the stream?
		rotate_barrier(dgree, 20, [chamber_height/2, chamber_width/2], obstacles)
		#re-introduce the removed obstacle
		ostacle_image = np.zeros((chamber_height, chamber_width,4), np.uint8)	# uint - 0-255 since rgba images... I have no idea why I thought I'd need 9, but now it is correct
		ostacle_image[obstacles[0],3] = 255
		image_obs = plt.imshow(ostacle_image, origin='lower', interpolation='none')# I don't want to have interpolation here
		for step in range(15):					# number of steps can also be an argument. for now, it is 20
			move_once(velocities, obstacles)
			rho, sum_vel_x, sum_vel_y=collide( velocities, relax_param, v0_horiz)
		the_fluid.set_array(plot_func(sum_vel_x, sum_vel_y, rho))
	#	print (dgree)
	#	if dgree>math.pi*8:
	#		exit()
		return (the_fluid, image_obs)		# return
else:
	def iter_func(arg):							# arg is probably the frame number - I on't need it, but this is how animate works - oh well
		global rho, sum_vel_x, sum_vel_y
		for step in range(20):					# number of steps can also be an argument. for now, it is 20
			move_once(velocities, obstacles)
			rho, sum_vel_x, sum_vel_y=collide( velocities, relax_param, v0_horiz)
		the_fluid.set_array(plot_func(sum_vel_x, sum_vel_y, rho))
		print(arg)
		return (the_fluid, image_obs)		# return



frames_needed=1200
#frames_needed=round(math.pi*3*100)
#let's animate!!
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='DoemenyB'))
animate = animation.FuncAnimation(this_figure, iter_func, frames=frames_needed,interval=1, blit=True)
#plt.show() #the appropriate should be commented out
animate.save(file_name+'.mp4', writer=writer)
