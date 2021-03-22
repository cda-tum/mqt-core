// i 0 1 2 3
// o 0 1 2 3
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
qreg r[2];
creg c[2];
creg d[2];

id q[0];
h q[0];
x q;
x q[0];
y q[0];
z q[0];
rx(pi/2) q[0];
rx(pi/4) q[0];
ry(pi/4) q[0];
ry(pi/2) q[0];
rz(-pi/8) q[0];
rz(-pi/0.7854) q[0];
s q[1];
sdg q[1];
U(0,0,-pi/4) q[1];
U(0,0,pi/4) q[0];

cx q[0],r[0];
cx q[0],r;
cx q,r[0];
cx q,r;

u3(1,2,3) r[0];
u2(1,2) r[1];
u1(1) q[0];
measure q -> c;
measure r[0] -> d[0];
measure q -> d;
