#!/usr/bin/python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#

### loading shell commands
#import os, os.path, sys
from math import floor
import numpy as np
#import scipy as sci
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



class PIC1D:
    def __init__(self,NGP=35,L=2.*np.pi/3.0600,dt=0.2,Number_Particles=30000,initial_velocity=0.2,
             initial_th_velocity=0.0,XP1=0.1,VP1=0.00,mode=1.0,cycles=2000,diag_step=100):
        self.NGP=NGP
        self.L=L
        self.dt=dt
        self.Number_Particles=Number_Particles
        self.initial_velocity=initial_velocity
        self.initial_th_velocity=initial_th_velocity
        self.XP1=XP1
        self.VP1=VP1
        self.mode=mode
        self.cycles=cycles
        ### constants
        self.WP=1.0
        self.QM=-1.0
        self.c=1.0
        self.Q=self.WP**2/(self.QM*Number_Particles/L)            # computational particle charge
        self.rho_back=-self.Q*Number_Particles/L            # background charge given by background (not moving) ions
        self.diag_step=diag_step
        self.weights=np.zeros((NGP,1))
        self.Efield=np.zeros((NGP,1))


#--- Initialize ---#        (moved to Constructor)
        self.dx = self.L/(1.*self.NGP-1)
        self.particle_position = np.linspace(0.,L,Number_Particles+1)[0:-1]
            #particle_position = np.linspace(0,L-L/Number_Particles,Number_Particles)
#--- THIS IS SPECIFIC TO THE TWO BEAM INSTABILITY WITH THE SAME SPECIES---#
        self.particle_velocity = initial_th_velocity * np.random.standard_normal((Number_Particles,))
        self.particle_velocity[range(0,Number_Particles-1,2)]=initial_velocity 
        self.particle_velocity[range(1,Number_Particles,2)]=-initial_velocity
        self.particle_velocity = np.divide( (self.particle_velocity+VP1*np.sin(2.*np.pi*self.particle_position/L*mode) ), (1.+self.particle_velocity*VP1*np.sin(2.*np.pi*self.particle_position/L*mode)/self.c**2))
#----RELATIVISTIC CORRECTION DISABLED RIGHT NOW(GAMMA NOT UPDATED IN MAIN CODE)
        #self.gamma = np.sqrt( (1./(1.-(self.particle_velocity/self.c)**2) ) )
        #self.u_particles=np.zeros((Number_Particles,))
        #for i in range(0,Number_Particles):
        #    self.u_particles[i] = self.gamma[i]*self.particle_velocity[i]
        for i in range(0,Number_Particles):
            self.particle_position[i] += XP1*(L/Number_Particles)*np.sin(2.*np.pi*self.particle_position[i]/L*mode);
            if self.particle_position[i]>=L:
                self.particle_position[i] -= L
            if self.particle_position[i] < 0:
                self.particle_position[i] += L         

                 
#--------------------------#
#--- charge deposition  ---#
    
    def particle_deposition(self): #,pos,dx,NGP):
        self.weights = np.zeros((NGP,1)) (NOW IN CONSTRUCTOR)
        for i in range(0,self.particle_position.size):
            v=floor(self.particle_position[i]/self.dx)
            self.weights[int(v)] += 1.-(self.particle_position[i]/self.dx-v)
            self.weights[int(v)+1] += self.particle_position[i]/self.dx-v

        self.weights[0]+=self.weights[-1] #periodic BC
        self.weights *= self.Q/self.dx
        self.weights += self.rho_back
    # return weights[0:NGP-1] no need to return as now in object.

#--------------------------#
#--- E-field calculator ---#
#--- finite difference scheme ---#
                      
    def E_calculator_potential(self): #rho,NGP,dx):
        NG=self.NGP-1
        
        self.weights *= self.dx**2  #rho->weights
        M=np.zeros((NG,NG))
        for i in range(0,NG):
            for j in range(0,NG):
                if i == j:
                    M[i,j]=+2.
                if i == j-1:
                    M[i,j]=-1.
                if i == j+1:
                    M[i,j]=-1.
        M[0,NG-1]=-1.0
        M[NG-1,0]=-1.0

        Phi=np.linalg.solve(M, self.weights[0:-1]) #electrostatic potential
        

    #Efield=np.zeros((NGP,1))   
        #finite difference gradient of potential is electric field
        for i in range(1,NG-1):
            self.Efield[i] = (Phi[i+1]-Phi[i-1]) / 2. / self.dx
        self.Efield[NG-1] = (Phi[0]-Phi[NG-2]) / 2. / self.dx
        self.Efield[0] = (Phi[1]-Phi[NG-1]) / 2. / self.dx
        self.Efield[NG]=self.Efield[0]
        self.Efield=-self.Efield
        self.Phi=np.append(Phi,Phi[0])
#     print M
#     print Phi
#      print self.Efield
#      exit(0)
    

        # return self.Efield ##NO LONGER NEEDED


#--- FEM approach --- (omit for now)
#def E_calculator_FEM(rho,NGP,dx):
#    NG=NGP-1
#    source=np.zeros((NG,1))
#    for i in range(0,NG):
#        if i == 0:
#            source[i] = +dx**2*(3./4.*rho[i]+1./8.*rho[NG-1]+1./8.*rho[i+1])
#        elif i == NG-1:
#            source[i] = +dx**2*(3./4.*rho[i]+1./8.*rho[i-1]+1./8.*rho[0])
#        else:
#            source[i] = +dx**2*(3./4.*rho[i]+1./8.*rho[i-1]+1./8.*rho[i+1])
#    M=np.zeros((NG,NG))
#    for i in range(0,NG):
#        for j in range(0,NG):
#            if i == j:
#                M[i,j]=+2.0
#            if j == i-1:
#                M[i,j]=-1.0
#            if j == i+1:
#                M[i,j]=-1.0
#       M[0,NG-1]=-1.0
#       M[NG-1,0]=-1.0
#      print M

#    Phi=np.linalg.solve(M, source)
    
    #--- E-field from Phi
#     self.Efield=np.zeros((NG,1))
#     for i in range(1,NG-2):
#         self.Efield[i] = (Phi[i+1]-Phi[i-1]) / 2. / dx
#     self.Efield[NG-1] = (Phi[0]-Phi[NG-2]) / 2. / dx
#     self.Efield[0] = (Phi[1]-Phi[NG-1]) / 2. / dx
#     self.Efield=-self.Efield

    #--- second technique - FEM approach
#    self.Efield=np.zeros((NGP,1))
#    for i in range(1,NG-1):
#        self.Efield[i] = (Phi[i+1]-Phi[i-1]) / dx
#    self.Efield[NG-1] = (Phi[0]-Phi[NG-2]) / dx
#    self.Efield[0] = (Phi[1]-Phi[NG-1]) / dx
#    self.Efield[NG]=self.Efield[0]
#    self.Efield=-self.Efield

#    return self.Efield
    
#--- pusher ---#  (THIS IS LEAPFROG)
    def velocity_pusher(self): #particle_position,particle_velocity,gamma,self.Efield,dx,dt):
        self.old_vel=self.particle_velocity
        for i in range(0,self.particle_velocity.size):
            v=np.floor(self.particle_position[i] /self.dx)
            w1= 1.-(self.particle_position[i]/self.dx-v)
            w2= 1.-w1
            self.particle_velocity[i] += self.QM * (w1*self.Efield[int(v)]+w2*self.Efield[int(v)+1])*self.dt
    #return self.particle_velocity, self.gamma  (Why gamma here?)

    def particle_pusher(self): #particle_position,particle_velocity,dt,L):
        for i in range(0,self.particle_position.size):
            self.particle_position[i] += self.particle_velocity[i]*self.dt
            if self.particle_position[i]>=self.L:
                self.particle_position[i] -= self.L
            if self.particle_position[i] < 0:
                self.particle_position[i] += self.L
    #return particle_position    

#--- Inputs ---#

    def diagnostics(self,plots=True,avg_vel=True,energy=True):
        if plots:
           fig = plt.figure(1, figsize=(6.0,6.0))
           ax1 = plt.subplot(411)
           ax1.plot(self.particle_position,self.particle_velocity,'om',ms=1.1)
           ax2 = plt.subplot(412)
           ax2.plot(np.linspace(0,self.L,self.weights.size),np.append(self.weights[0:-1],self.weights[0]),'k')
           ax3 = plt.subplot(413)
           ax3.plot(np.linspace(0,self.L,self.Efield.size),self.Efield,'k')
           fv=gaussian_kde(self.particle_velocity)
           pts=np.linspace(-1,1,1000)
           ax4 = plt.subplot(414)
           ax4.plot(pts,fv.evaluate(pts))
           plt.show()
           plt.show()
           
        if avg_vel:
            if hasattr(self,'avg_vel'):
                self.avg_vel=np.append(self.avg_vel,np.mean(self.particle_velocity))
            else:
                self.avg_vel=np.array(np.mean(self.particle_velocity))
        if energy:
             kinetic_energy=np.sum(self.old_vel*self.particle_velocity)
             if hasattr(self,'kin_eng'):
                 self.kin_eng=np.append(self.kin_eng,kinetic_energy)
             else:
                 self.kin_eng=np.array(kinetic_energy)
             potential_energy=.5*self.dx*np.sum(self.Phi[0:-1]*self.weights[0:-1])  #comp trap rule for periodic data
             if hasattr(self,'pot_eng'):
                 self.pot_eng=np.append(self.pot_eng,potential_energy)
             else:
                 self.pot_eng=np.array(potential_energy)                     
           




