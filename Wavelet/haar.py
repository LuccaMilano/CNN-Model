import math
from numpy import *

import matplotlib
import matplotlib.pyplot as plt

## This function plots the signal flow graph (SFG) for a matrix chain. The matrices are assumed to be passed as argument in the right order of multiplication. Furthermore, the matrices are supposed to be square, i.e. NxN, and of same order N. 
# @param matrixChain
# a list of Numpy matrices 
# @param filename
# (optional) the output name (with extension) for saving the plotted SFG
def plotSFG(matrixChain, filename=None):
   matplotlib.rc('text', usetex=False)
   #matplotlib.rcParams['text.usetex'] = True
   #matplotlib.rcParams['text.latex.unicode'] = True
   
   nMatrices = len(matrixChain)
   matrixChain = matrixChain[::-1]
   
   curMatrix = matrixChain[0]
   N = curMatrix.shape[0] 
   matrixCounter = 0
   
   plt.cla()
   ax = plt.axes()
   ax.axis('off')
   
   x = 1
   ys = array(range(N,0,-1))
   
   while matrixCounter < nMatrices:
      curMatrix = matrixChain[matrixCounter]
      ax.scatter(matrixCounter*ones(N), ys, s = 15, c = 'r', alpha = .75)
      for i in range(N):
         for j in range(N):
            curValue = curMatrix[i,j].astype(float)
            if curValue != 0:
               style = None
               if curValue < 0:
                  style = 'dashed'
               if abs(curValue) != 1:
                  if (curValue).is_integer():
                     text = str(int(abs(curValue)))
                  else:
                     text = str(round(abs(curValue),2))
                  ax.annotate(text, xy=(matrixCounter+.1, (ys[j]*.25+ys[i]*.75)+.025))
               ax.annotate("", xy=(matrixCounter+1, ys[j]), xycoords='data', xytext=(matrixCounter, ys[i]), textcoords='data', arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", ls=style))
      matrixCounter+=1
   
   for i in range(N):
      ax.annotate(r'${\mathbf{v}_{'+str(i)+'}}$', xy=(0-.0625, ys[i]), xycoords='data', ha='right')#, usetex=True)
      ax.annotate(r'${\mathbf{w}_{'+str(i)+'}}$', xy=(matrixCounter+.03125, ys[i]), xycoords='data')#, usetex=True)
      
   ax.scatter(matrixCounter*ones(N), ys, s = 15, c = 'r', alpha = .75)
   
   if filename:
      plt.savefig(filename, bbox_inches='tight')
   plt.show()
   
############################ TESTS ############################

'''
N = 128 
v = matrix(array(range(N))).transpose()
v2 = matrix(array(range(N*N)).reshape(N,N))
'''

#N = 4
#v = matrix(array([9,7,3,5])).transpose()
#v2 = matrix([[9,7,3,5],[9,7,3,5],[9,7,3,5],[9,7,3,5]])



############################ TEST FOR SECTION 1.1 ############################

# Equation 1
def H(N, f): 
  return(f*T(N))

# Equation 2
def T(N):
  return(concatenate([C(N),D(N)]))

# Equation 3
def C(N):
  z = zeros(N*(N//2)).reshape(N//2,N)
  for i in range(0,N,2):
    z[i//2,i] = 1
    z[i//2,i+1] = 1
  return(z)

# Equation 4
def D(N):
  z = zeros(N*(N//2)).reshape(N//2,N)
  for i in range(0,N,2):
    z[i//2,i] = 1
    z[i//2,i+1] = -1
  return(z)

#f = 1/2.

#print(H(N,f))
#print(T(N))
#print(C(N))
#print(D(N))

#w = dot(H(N,f),v) # Equation 5
#print w

#v = dot(linalg.inv(H(N,f)),w) # Equation 7
#print v

#w2 = dot(dot(H(N,f),v2),linalg.inv(H(N,f))) # Equation 6
#print w2

#v2 = dot(dot(linalg.inv(H(N,f)),w2),H(N,f)) # Equation 8
#print v2


#f = 1/sqrt(2.)

#print(H(N,f))
#print(T(N))
#print(C(N))
#print(D(N))

#w = dot(H(N,f),v) # Equation 5
#print w

#v = dot(linalg.inv(H(N,f)),w) # Equation 7
#print v
#v = dot(H(N,f).transpose(),w) # Equation 7
#print v

#w2 = dot(dot(H(N,f),v2),linalg.inv(H(N,f))) # Equation 6
#print w2

#v2 = dot(dot(linalg.inv(H(N,f)),w2),H(N,f)) # Equation 8
#print v2

#w2 = dot(dot(H(N,f),v2),H(N,f).transpose()) # Equation 6
#print w2

#v2 = dot(dot(H(N,f).transpose(),w2),H(N,f)) # Equation 8
#print v2



############################ TEST FOR SECTION 1.2.1 ############################

def Hl(N,l,f): # Equation 9
  i = identity(N)
  matrices = []
  for k in range(int(math.log(N,2))+1-l,int(math.log(N,2))-1+1): # It is necessary to add 1 in Python range functions
    i = dot(i,H_(N,2**k,f))
    matrices += [H_(N,2**k,f)]
  matrices += [H(N,f)]
  #plotSFG(matrices)
  return(dot(i,H(N,f)))

def H_(N,n,f): # Equation 10
  Z = zeros((N-n)*(n)).reshape(N-n,n)
  left = concatenate([H(n,f), Z])
  right = concatenate([Z.transpose(), identity(N-n)])
  return(concatenate([left,right],axis=1))

def S(N, Hl, f):
  if f == 1/sqrt(2): # Equation 12
    return(sqrt(linalg.inv(dot(Hl,Hl.transpose()))))	
  elif f == 1/2.: # Equation 13
    return(linalg.inv(dot(Hl,Hl.transpose())))

'''
f = 1/2.

#print(H_(N,N/2,f))
hln1 = Hl(N,int(log2(N)),f)
#print(hln1)
'''

#s1 = S(N,(Hl(N,2,1)), f) # Equation 13
#print(s1)
#hln2 = dot(s1,Hl(N,2,1)) # Equation 11
#print(hln2)
#print(allclose(hln1,hln2)) # Equation 11 - Confirmation

'''
w = dot(hln1,v) # Equation 16
print w

v = dot(linalg.inv(hln1),w) # Equation 18
print v

w2 = dot(dot(hln1,v2),linalg.inv(hln1)) # Equation 17
print w2

v2 = dot(dot(linalg.inv(hln1),w2),hln1) # Equation 19
print v2
'''

'''
f = 1/sqrt(2.)

#print(H_(N,N/2,f))
hln1 = Hl(N,int(log2(N)),f)
#print(hln1)
'''

#s2 = S(N,(Hl(N,2,1)), f) # Equation 12
#print(s2)
#hln2 = dot(s2,Hl(N,2,1)) # Equation 11
#print(hln2)
#print(allclose(hln1,hln2)) # Equation 11 - Confirmation

'''
w = dot(hln1,v) # Equation 16
print w

v = dot(linalg.inv(hln1),w) # Equation 18
print v
v = dot(hln1.transpose(),w) # Equation 18
print v

w2 = dot(dot(hln1,v2),linalg.inv(hln1)) # Equation 17
print w2

v2 = dot(dot(linalg.inv(hln1),w2),hln1) # Equation 19
print v2

w2 = dot(dot(hln1,v2),hln1.transpose()) # Equation 17
print w2

v2 = dot(dot(hln1.transpose(),w2),hln1) # Equation 19
print v2
'''

############################ TEST FOR SECTION 1.2.2 ############################

def Hlp(N,l,f): # Equation 14
  i = identity(N)
  matrices = []
  for k in range(int(math.log(N,2))+1-l,int(math.log(N,2))-1+1): # It is necessary to add 1 in Python range functions
    i = dot(i,Hp_(N,2**k,f))
    matrices += [Hp_(N,2**k,f)]
  matrices += [H(N,f)]
  #plotSFG(matrices)
  return(dot(i,H(N,f)))

def Hp_(N,n,f): # Equation 15
  Z = zeros(N*N).reshape(N,N)
  times = N/n
  for i in range(times):
    Z[n*i:n*(i+1),n*i:n*(i+1)] = H(n,f)
  return(Z)


'''
f = 1/2.

#print(Hp_(N,N/2,f))
hln1 = Hlp(N,int(log2(N)),f)
#print(hln1)
'''

#s1 = S(N,(Hlp(N,2,1)), f) # Equation 13
#print(s1)
#hln2 = dot(s1,Hlp(N,2,1)) # Equation 11
#print(hln2)
#print(allclose(hln1,hln2)) # Equation 11 - Confirmation

'''
w = dot(hln1,v) # Equation 16
print w

v = dot(linalg.inv(hln1),w) # Equation 18
print v

w2 = dot(dot(hln1,v2),linalg.inv(hln1)) # Equation 17
print w2

v2 = dot(dot(linalg.inv(hln1),w2),hln1) # Equation 19
print v2
'''
'''
f = 1/sqrt(2.)

#print(H_(N,N/2,f))
hln1 = Hlp(N,int(log2(N)),f)
#print(hln1)
'''

#s2 = S(N,(Hlp(N,2,1)), f) # Equation 12
#print(s2)
#hln2 = dot(s2,Hlp(N,2,1)) # Equation 11
#print(hln2)
#print(allclose(hln1,hln2)) # Equation 11 - Confirmation

'''
w = dot(hln1,v) # Equation 16
print w

v = dot(linalg.inv(hln1),w) # Equation 18
print v
v = dot(hln1.transpose(),w) # Equation 18
print v

w2 = dot(dot(hln1,v2),linalg.inv(hln1)) # Equation 17
print w2

v2 = dot(dot(linalg.inv(hln1),w2),hln1) # Equation 19
print v2

w2 = dot(dot(hln1,v2),hln1.transpose()) # Equation 17
print w2

v2 = dot(dot(hln1.transpose(),w2),hln1) # Equation 19
print v2
'''

"""
#PACKET 2D 2-level transform over Lena image

import scipy.misc
from matplotlib import pyplot

f = 1/sqrt(2)

lena = scipy.misc.lena()
pkt2d2l = Hlp(lena.shape[0],2,f)

wlena = dot(dot(pkt2d2l,lena),pkt2d2l.transpose())

pyplot.gray()
pyplot.imshow(wlena)
pyplot.show()
"""
