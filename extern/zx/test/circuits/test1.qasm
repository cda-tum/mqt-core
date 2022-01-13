OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
// u3(3.141592653589793,0.0,3.141592653589793) q[2];
