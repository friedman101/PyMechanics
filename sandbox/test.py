import numpy as np
import sympy as sp
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

t = sp.symbols('t')
theta1 = sp.Function("theta1")(t)
theta2 = sp.Function("theta2")(t)
l1, l2, psi2 = sp.symbols("l1 l2 psi2")

q = sp.Matrix([theta1,theta2])

R1 = sp.Matrix([[1,0,0],[0,sp.cos(theta1),-sp.sin(theta1)],[0,sp.sin(theta1),sp.cos(theta1)]])

R2a = sp.Matrix([[1,0,0],[0,sp.cos(theta2),-sp.sin(theta2)],[0,sp.sin(theta2),sp.cos(theta2)]])
R2b = sp.Matrix([[sp.cos(psi2),-sp.sin(psi2),0],[sp.sin(psi2),sp.cos(psi2),0],[0,0,1]])
R2 = R1*R2b*R2a

x1 = R1*sp.Matrix([[0],[0],[-l1/2]])
x2 = R1*sp.Matrix([[0],[0],[-l1]]) + R2*sp.Matrix([[0],[0],[-l2/2]])

Jv1 = x1.jacobian(q)
Jv2 = x2.jacobian(q)

tmp = R1.diff(theta1)*R1.transpose()
a1 = tmp[2,1]
a2 = tmp[0,2]
a3 = tmp[1,0]
Jw1q1 = sp.Matrix([[a1],[a2],[a3]])

tmp = R1.diff(theta2)*R1.transpose()
a1 = tmp[2,1]
a2 = tmp[0,2]
a3 = tmp[1,0]
Jw1q2 = sp.Matrix([[a1],[a2],[a3]])

Jw1 = Jw1q1.row_join(Jw1q2)

tmp = R2.diff(theta1)*R2.transpose()
a1 = tmp[2,1]
a2 = tmp[0,2]
a3 = tmp[1,0]
Jw2q1 = sp.Matrix([[a1],[a2],[a3]])

tmp = R2.diff(theta2)*R2.transpose()
a1 = tmp[2,1]
a2 = tmp[0,2]
a3 = tmp[1,0]
Jw2q2 = sp.Matrix([[a1],[a2],[a3]])

Jw2 = Jw2q1.row_join(Jw2q2)

Jv = Jv1.col_join(Jv2)
Jw = Jw1.col_join(Jw2)

J = Jv.col_join(Jw)
Jdot = J.diff(t)

theta1_dot = sp.symbols("theta1_dot")
theta2_dot = sp.symbols("theta2_dot")
Jdot = Jdot.subs(theta1.diff(t), theta1_dot)
Jdot = Jdot.subs(theta2.diff(t), theta2_dot)


def McFunc (m1, I1, m2, I2):
	blk1 = np.eye(3)*m1
	blk3 = I1
	blk2 = np.eye(3)*m2
	blk4 = I2
	return block_diag(blk1, blk2, blk3, blk4)

array2mat = [{'ImmutableMatrix': np.matrix}, 'numpy']
Jwfunc = sp.lambdify((theta1,theta2,l1,l2,psi2), Jw, modules=array2mat)
Jvfunc = sp.lambdify((theta1,theta2,l1,l2,psi2), Jv, modules=array2mat)
Jfunc = sp.lambdify((theta1,theta2,l1,l2,psi2), J, modules=array2mat)
JdotFunc = sp.lambdify((theta1, theta1_dot, theta2, theta2_dot,l1,l2,psi2), Jdot, modules=array2mat)

Mc = McFunc(10, 10*np.eye(3), 10, 10*np.eye(3))

dt = 0.01
t0 = 0
tend = 50
t = np.arange(t0, tend, dt)

q = np.matrix([0,np.pi/8])
qdot = np.matrix([0,0])
for i in range(1, t.shape[0]-1):
	print i
	theta1 = q[i-1,0]
	theta2 = q[i-1,1]
	theta1_dot = qdot[i-1,0]
	theta2_dot = qdot[i-1,1]
	myqdot = np.matrix([[theta1_dot],[theta2_dot]])
	l1 = 5
	l2 = 7
	
	psi2 = 0*np.pi/180.0

	f = np.matrix([[0],[0],[-9.81],[0],[0],[-9.81]])
	tau = np.zeros((6,1))

	J = Jfunc(theta1, theta2, l1, l2, psi2)
	Jw = Jwfunc(theta1, theta2, l1, l2, psi2)
	Jv = Jvfunc(theta1, theta2, l1, l2, psi2)
	Jdot = JdotFunc(theta1, theta1_dot, theta2, theta2_dot, l1, l2, psi2)

	tmp =  Jw*myqdot
	a1 = tmp[0,0]
	a2 = tmp[1,0]
	a3 = tmp[2,0]
	b1 = tmp[3,0]
	b2 = tmp[4,0]
	b3 = tmp[5,0]

	tmp2 = np.matrix([[0,-a3,a2],[a3,0,-a1],[-a2,a1,0]])
	tmp3 = np.matrix([[0,-b3,b2],[b3,0,-b1],[-b2,b1,0]])
	tmp4 = block_diag(tmp2, tmp3)
	myzeros = np.zeros((6,6))
	omegaWiggle = np.vstack((np.hstack((myzeros, myzeros)),np.hstack((myzeros, tmp4))))

	M = J.transpose()*Mc*J
	C = (J.transpose()*Mc*Jdot + J.transpose()*omegaWiggle*Mc*J)*myqdot
	Q = Jv.transpose()*f + Jw.transpose()*tau

	q_ddot = np.linalg.inv(M)*(Q-C)

	theta1 = theta1 + theta1_dot*dt
	theta2 = theta2 + theta2_dot*dt
	theta1_dot = theta1_dot + q_ddot[0,0]*dt
	theta2_dot = theta2_dot + q_ddot[1,0]*dt

	q = np.vstack((q, np.matrix([theta1,theta2])))
	qdot = np.vstack((qdot, np.matrix([theta1_dot,theta2_dot])))


plt.plot(q)
plt.show()