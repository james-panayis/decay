import ROOT
import numpy
import pandas
import math
import random

c = 299792458

B0 = {
"P": [0,0,50*(10**3)],
"x": [0,0,0],
"m": [5279.66]}

#B0Vz = B0["P"][2] / B0["m"][0]
#beta = B0Vz / c
#gamma = 1/(math.sqrt(1 - (beta**2)))

Pi = {
"P": [0,0,0],
"x": [0,0,0],
"m": [139.57039]}

K = {
"P": [0,0,0],
"x": [0,0,0],
"m": [493.667]}

Eb = B0["m"][0]*(c**2)

a = Pi["m"][0]
b = K["m"][0]
d = B0["m"][0]

P = (c * math.sqrt((a**4) + (b**4) + (d**4) - 2*(a**2)*(b**2) - 2*(a**2)*(d**2) - 2*(b**2)*(d**2)))/(2*d)


print("P: ", P)
print("Energy of the B0: ", Eb, " MeV")

PiV = P / Pi["m"][0]
KV = -P / K["m"][0]

print("PiV", PiV)

print("KV", KV)

# Is transverse momentum 0 as isotropic? evenly distributed up and down?

PiV = math.sqrt(P**2/(Pi["m"][0]**2 + P**2/c**2))

KV = math.sqrt(P**2/(K["m"][0]**2 + P**2/c**2))

print("PiV: ", PiV)

print("KV: ", KV)

i=0

#while(i<100000):
#    ran = numpy.random.normal(0)













