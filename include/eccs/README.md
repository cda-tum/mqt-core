# Error-correcting codes
Below, every available error-correcting code is shortly explained. 

*Draft version: currently only key notes!*

## Shor codes ([source](https://link.aps.org/doi/10.1103/PhysRevA.52.R2493))
### Q3Shor
chosen: The Q3Shor code was the first proposed quantum error-correcting code. Although it does not detect phase flips, it is relatively easy to understand. 

"repetition" code (no-cloning), logical 0/1 mapped to 000 / 111. Measuring via syndromes. 
### Q9Shor
chosen: The Q9Shor code can be seen as a "two-level Q3Shor code". In contrast to the basic Q3Shor code, it can also detect phase flips. Thus, arbitrary one-qubit errors can be detected.  

## Stabilizer codes
### Q5Laflamme ([source](https://link.aps.org/doi/10.1103/PhysRevLett.77.198))
chosen: Although it is not easy to support more complicated operations for the Q5Laflamme code, it has been chosen as it is the smallest error-correcting code which can detect arbitrary one-qubit errors. 

### Q7Steane ([source](https://link.aps.org/doi/10.1103/PhysRevLett.77.793))
chosen: The Q7Steane code is structured in a way such that many qubit operations as well as simple detection and correction are well-supported.

## Surface Codes ([source](https://doi.org/10.48550/arXiv.quant-ph/9811052), [source](https://doi.org/10.48550/arXiv.quant-ph/0110143))
chosen: Surface codes are the state of the art in quantum circuit research. 
In contrast to the previous kinds of ECCs, surface codes can be extended easily without changing the concept of the correction mechanism. 
Moreover, the qubits only have to interact with few neighboring qubits, which makes these types of ECCs serializable in hardware more easily. 

### Q9Surface
chosen: The Q9Surface code is the smallest possible surface code which can detect and correct arbitrary one-qubit errors. 
### Q18Surface
chosen: The Q18-Surface code is the smallest possible surface code we found that can easily support a Hadamard operation. 
