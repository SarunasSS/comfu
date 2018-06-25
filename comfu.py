import numbers
import numpy as np
import itertools

APPROX_ORDER = 5

class Function():

    def __init__( self, n_inputs=None, n_outputs=None, A=None ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if A is None:
            self.A = np.random.random( [ APPROX_ORDER ] * n_inputs + [ n_outputs ] ) * 2.0 - 1.0
        else:
            assert ( n_inputs is None or n_inputs == A.ndim-1 ) and ( n_outputs is None or n_outputs == A.shape[ -1 ] )
            self.A = A
            self.n_inputs = A.ndim - 1
            self.n_outputs = A.shape[ -1 ]

    def __call__( self, x ):
        assert x.shape[ 1 ] == self.n_inputs
        n_batch = x.shape[ 0 ]

        Y = self.A.copy()
        Y = np.moveaxis( Y, -1, 0 )
        Y = np.expand_dims( Y, 0 )
        Y = np.expand_dims( Y, -1 )
        # Y.shape = batch x n_outputs x approx_order_1 x ... x approx_order_n_inputs x 1 

        for dim in xrange( self.n_inputs ):
            X = np.concatenate( [ np.power( x[ :, [ dim ] ], p ) for p in xrange( self.A.shape[ dim ] ) ], axis=1 )[ :, None, :, None  ]
            for i in xrange( self.n_inputs-1 ):
                X = np.expand_dims( X, -3 )
            # X.shape = batch x n_outputs x 1^(n_inputs-1) x approx_order x 1
            Y = np.moveaxis( Y, dim+2, -1 )
            Y = np.matmul( Y, X )
            Y = np.moveaxis( Y, -1, dim+2 )

        for i in xrange( self.n_inputs+1 ):
            Y = Y.squeeze( -1 )

        return Y

    def __add__( self, another ):
        if isinstance( another, numbers.Number ):
            new_A = self.A.copy()
            new_A[ ( 0, ) * self.n_inputs ] += another
        else:
            assert self.n_inputs == another.n_inputs
            assert self.n_outputs == another.n_outputs
            new_A = self.A + another.A

        return Function( A=new_A )

    def __sub__( self, another ):
        if isinstance( another, numbers.Number ):
            new_A = self.A.copy()
            new_A[ ( 0, ) * self.n_inputs ] -= another
        else:
            assert self.n_inputs == another.n_inputs
            assert self.n_outputs == another.n_outputs
            new_A = self.A - another.A

        return Function( A=new_A )

    def __mul__( self, another ):
        if isinstance( another, numbers.Number ):
            new_A = self.A * another
        else:
            #assert self.n_inputs == 1
            assert self.n_inputs == another.n_inputs
            #assert self.n_outputs == 1
            assert self.n_outputs == another.n_outputs
            A = self.A
            another_A = another.A

            new_A = np.zeros( [ a+b-1 for ( a, b ) in zip( A.shape[:-1], another_A.shape[:-1] ) ] + [ self.n_outputs ] ) 
            for o_dim in xrange( self.n_outputs ):
                A_sub = np.take( A, o_dim, -1 )
                another_A_sub = np.take( another_A, o_dim, -1 )
                
                # A_sub_prod = np.einsum( 'kl,mn->klmn', A_sub, another_A_sub )
                A_sub_prod = np.tensordot( A_sub[ None, ], another_A_sub[ None, ], [ [ 0 ], [ 0 ] ] )

                for idx, val in np.ndenumerate( A_sub_prod ):
                    new_degrees = tuple( [ sum( idx[ input_i::n_inputs ] ) for input_i in xrange( self.n_inputs ) ] )
                    new_A[ new_degrees + ( o_dim, ) ] += val

        return Function( A=new_A )

    def xmul( self, value, axis ):

        degrees = np.arange( self.A.shape[ axis ] )
        slc = [ None if a != axis else slice( None ) for a in xrange( self.A.ndim ) ]
        values = np.power( np.array( value ), degrees )[ slc ]
        new_A = self.A * values
        
        return Function( A=new_A )

    def addim( self, axis, order=APPROX_ORDER ):

        new_A_shape = list( self.A.shape )
        new_A_shape.insert( axis, 1 )

        new_A = np.zeros( new_A_shape )
        slc = [ slice( None ) if a != axis else 0 for a in xrange( new_A.ndim ) ]
        new_A[ slc ] = self.A

        return Function( A=new_A )


    def dx( self, axis=None ):
        if axis is None: 
            assert self.n_inputs == 1
            axis = 0 

        new_A = self.A.copy()
        
        factors = np.arange( self.A.shape[ axis ] )
        slc = [ None if a != axis else slice( None ) for a in xrange( new_A.ndim ) ]
        new_A *= factors[ slc ]

        slc = [ slice( None ) if a != axis else slice( 1, None, None ) for a in xrange( new_A.ndim ) ]
        new_A = new_A[ slc ]
        
        return Function( A=new_A )

    def int( self, axis=None ):    
        if axis is None: 
            assert self.n_inputs == 1
            axis = 0 

        new_A = self.A.copy() 

        factors = np.arange( 1, self.A.shape[ axis ] + 1  )
        slc = [ None if a != axis else slice( None ) for a in xrange( new_A.ndim ) ]
        new_A /= factors[ slc ]

        # new_A /= np.arange( 1, new_A.shape[ 0 ]+1 )
        new_A = np.insert( new_A, 0, 0, axis )

        return Function( A=new_A )

if True:
    for n_inputs in range( 1, 3 ):
        for n_outputs in range( 1, 3 ):
            # Tests
            n_inputs = 3
            n_outputs = 2

            f = Function( n_inputs=n_inputs, n_outputs=n_outputs )
            g = Function( n_inputs=n_inputs, n_outputs=n_outputs )

            x = np.random.random( [ 10, n_inputs ] )
            # x = np.arange( 10+1 )[ :, None ] / 10.0

            # Tests
            y = f( x )
            assert y.ndim == 2 and y.shape[ 0 ] == x.shape[ 0 ] and y.shape[ 1 ] == n_outputs
            assert np.allclose( f( x ) + 1, ( f + 1 )( x ) ) 
            assert np.allclose( f( x ) - 2, ( f - 2 )( x ) ) 
            assert np.allclose( f( x ) * 3, ( f * 3 )( x ) ) 
            assert np.allclose( f( x ) + g( x ), ( f + g )( x ) ) 
            assert np.allclose( f( x ) - g( x ), ( f - g )( x ) ) 
            assert np.allclose( f( x ) * g( x ), ( f * g )( x ) ) 
            assert np.allclose( f.int( 0 ).dx( 0 )( x ), f( x ) )

            x_bar = x.copy()
            x_bar[ :, 0 ] *= 3
            y_bar = f( x_bar )
            assert np.allclose( f.xmul( 3, axis=0 )( x ), y_bar ) 

            x_bigger = np.random.random( [ 10, n_inputs + 1 ] )
            f.addim( 1 )( x_bigger )

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    n_inputs = 1
    n_outputs = 1

    f = Function( n_inputs=n_inputs, n_outputs=n_outputs )
    g = Function( n_inputs=n_inputs, n_outputs=n_outputs )

    x = np.sort( np.random.random( [ 10, n_inputs ] ) - 0.5, 0 )

    plt.plot( x.squeeze(), f( x ).squeeze(), label='f' )
    plt.plot( x.squeeze(), g( x ).squeeze(), label='g' )
    plt.plot( x.squeeze(), ( f + g )( x ).squeeze(), label='f+g' )
    plt.plot( x.squeeze(), ( f - g )( x ).squeeze(), label='f-g' )
    plt.plot( x.squeeze(), ( f * g )( x ).squeeze(), label='f*g' )
    plt.plot( x.squeeze(), f.dx()( x ).squeeze(), label='dfdx' )
    plt.plot( x.squeeze(), f.int()( x ).squeeze(), label='integral f' )
    plt.plot( x.squeeze(), f.xmul( 0.5, 0 )( x ).squeeze(), label='int f( 0.5 * x )' )

    plt.legend()

    plt.show()




