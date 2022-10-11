import ROOT
import numpy
import pandas
import math
import random

c = 299792458

B0 = {
"P": [0,0,50*c*(10**3)],
"m": [5279.66]}

Pi = {
"P": [0,0,0],
"m": [139.57039]}

K = {
"P": [0,0,0],
"m": [493.667]}

#Energy of B0
Eb = B0["m"][0]*(c**2)

a = Pi["m"][0]
b = K["m"][0]
d = B0["m"][0]

#Equation for average momentum from conservation of momentum law
P = (c * math.sqrt((a**4) + (b**4) + (d**4) - 2*(a**2)*(b**2) - 2*(a**2)*(d**2) - 2*(b**2)*(d**2)))/(2*d)

#print("P: ", P)
#print("Energy of the B0: ", Eb, " MeV")

#Particle Velocities modulus
B0V = math.sqrt(B0["P"][2]**2/(B0["m"][0]**2 + B0["P"][2]**2/c**2))
PiV = math.sqrt(P**2/(Pi["m"][0]**2 + P**2/c**2))
KV = math.sqrt(P**2/(K["m"][0]**2 + P**2/c**2))
#print("B0V: ", B0V)
#print("PiV: ", PiV)
#print("KV: ", KV)

#B distance before decay
B_ml_bf = 1519*10**(-15)
B_gamma = 1/math.sqrt(1-(B0V**2/c**2))
B_ml_df = B_gamma * B_ml_bf
B_dist = B_ml_df * B0V
#print(B_dist)


i=0
PiPT = 0
KPT = 0
PiPav = 0
KPav = 0
Pid = 0
Kd = 0
count = 1000000

while(i<count):

    #Random distribution from a Gaussian for positions on a sphere
    ranx = numpy.random.normal(0)
    rany = numpy.random.normal(0)
    ranz = numpy.random.normal(0)

    Pixv = (ranx / math.sqrt(ranx**2 + rany**2 +ranz**2))*PiV
    Piyv = (rany / math.sqrt(ranx**2 + rany**2 +ranz**2))*PiV
    Pizv = (ranz / math.sqrt(ranx**2 + rany**2 +ranz**2))*PiV

    Kxv = (-ranx / math.sqrt(ranx**2 + rany**2 +ranz**2))*KV
    Kyv = (-rany / math.sqrt(ranx**2 + rany**2 +ranz**2))*KV
    Kzv = (-ranz / math.sqrt(ranx**2 + rany**2 +ranz**2))*KV

    #print(Pixv, Piyv, Pizv)
    #print("MOD: ", math.sqrt(Pixv**2 + Piyv**2 + Pizv**2))
    #print(Kxv, Kyv, Kzv)
    #print("MOD: ", math.sqrt(Kxv**2 + Kyv**2 + Kzv**2))


    #Making Meson Velocities
    Pixu = (math.sqrt(1- B0V**2/c**2)*Pixv)/(1+(B0V/c**2)*Pizv)
    Piyu = (math.sqrt(1- B0V**2/c**2)*Piyv)/(1+(B0V/c**2)*Pizv)
    Pizu = (Pizv + B0V)/(1 + (B0V/c**2)*Pizv)

    Kxu = (math.sqrt(1- B0V**2/c**2)*Kxv)/(1+(B0V/c**2)*Kzv)
    Kyu = (math.sqrt(1- B0V**2/c**2)*Kyv)/(1+(B0V/c**2)*Kzv)
    Kzu = (Kzv + B0V)/(1 + (B0V/c**2)*Kzv)
   
    #Printing Meson Velocities 
    #print("Pi: ", Pixu, Piyu, Pizu, " Mag: ", math.sqrt(Pixu**2+Piyu**2+Pizu**2))
    #print("K: ", Kxu, Kyu, Kzu, " Mag: ", math.sqrt(Kxu**2+Kyu**2+Kzu**2))

    #Transverse and average momentum
    PiuT = math.sqrt(Pixu**2 + Piyu**2)
    KuT = math.sqrt(Kxu**2 + Kyu**2)

    Piu = math.sqrt(Pixu**2 + Piyu**2 + Pizu**2)
    Ku = math.sqrt(Kxu**2 + Kyu**2 + Kzu**2)

    #Creating momentum in each axis
    PiPav += 1/(math.sqrt(1-(Piu**2/c**2)))*Pi["m"][0]*Piu
    PiPT += 1/(math.sqrt(1-(Piu**2/c**2)))*Pi["m"][0]*PiuT

    KPav += 1/(math.sqrt(1-(Ku**2/c**2)))*K["m"][0]*Ku
    KPT += 1/(math.sqrt(1-(Ku**2/c**2)))*K["m"][0]*KuT

    #Impact Param calculation
    Pid += B_dist * math.sqrt(Pixu**2 + Piyu**2)/ math.sqrt(Pixu**2 + Piyu**2 + Pizu**2)
    Kd += B_dist * math.sqrt(Kxu**2 + Kyu**2)/ math.sqrt(Kxu**2 + Kyu**2 + Kzu**2)
 
    i+=1

#Average PT and P
FinalPiPT = (PiPT / count)/(c)
FinalKPT = (KPT / count)/(c)
FinalPiP = (PiPav / count)/(c)
FinalKP = (KPav / count)/(c)

print("Average Pi PT: ", FinalPiPT)
print("Average K PT: ", FinalKPT)
print("Average Pi momentum: ", FinalPiP)
print("Average K momentum: ", FinalKP)

Pidav = Pid/count
Kdav = Kd/count

print("Pi impact param av: ", Pidav)
print("K impact param av: ", Kdav)


