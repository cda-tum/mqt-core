# declare variables names
.v a b c d
# declare inputs (missing variables are assumed ancillaries)
.i a b c
# declare outputs (missing variables are assumed garbage)
.o a b c d
# declare constants
.c 1
# begin circuit description
BEGIN
H a
X b
Y c
Z d
S a
S* b
P c
P* d
T a
T* b
Rx(2.0) c
Ry(pi/2^3) d
Rz(-3.2e-2) a
cnot b c
tof d a b
Zd c d a
swap b c
X d a b c
END
