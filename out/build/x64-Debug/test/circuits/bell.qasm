OPENQASM 2.0;
//include "qelib1.inc";
qreg q[2];
U(pi/2,0,pi) q[0];
CX q[0],q[1];
