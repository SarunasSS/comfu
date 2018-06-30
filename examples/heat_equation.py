import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.append( '../' )

from comfu import Function

# np.random.seed( 12 )

# Parameters
dt = 0.1 
T = 100
alpha = 1

# Boundary function conditions
# boundary_condition = Function( n_inputs=1, n_outputs=1 )
boundary_condition = Function( A=np.array( [[ 0, 0, 1 ]] ).T )

# The actual solve

x = np.linspace( -1, 1, 100 )[ :, None ]

plt.figure()
plt.plot( x, boundary_condition( x ), 'r' )

u = boundary_condition
for t in np.arange( 0, T, dt ):
	if t / dt % 100 == 0:
		plt.plot( x, u( x ), 'b', alpha=t/float( T ) )
		print t/float( T )
	u = u + u.dx( 0 ).dx( 0 ) * alpha * dt 
plt.show()