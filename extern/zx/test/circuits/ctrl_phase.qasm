OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];

cp(pi/8) q[0],q[1];
cp(-pi/8) q[0],q[1];

