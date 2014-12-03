import numpy

"""
exex : EXafs EXtraction
       python routines to extract the EXAFS signal ( chi(k) ) from the absorption 
       spectrum ( mu(E) ) 


       dependencies: numpy (matplotlib may be used for plotting, but not required)

       MODIFICATION HISTORY:
           20141204 srio@esrf.eu, written
"""

__author__ = "Manuel Sanchez del Rio"
__contact__ = "srio@esrf.eu"
__copyright__ = "ESRF, 2014"


#
#
#


##
## definition: "set" is a numpy array (npoints,ncols) containing spetral data
##

def plotSet(set0,over=None,pltOk=True,xtitle="x",ytitle="y",toptitle="",label="",xmin=None,xmax=None,ymin=None,ymax=None):
    r"""
        plotSet(set0,over=None,pltOk=True,xtitle="x",ytitle="y",toptitle="",label="")
    """

    if pltOk:
        plt.figure(0)

        plt.title(toptitle)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)

 
        if label != "":
            ax = plt.subplot(111)
            ax.legend(bbox_to_anchor=(1.1, 1.05))
            plt1 = plt.plot(set0[:,0],set0[:,1],'green')
            if over != None:
                plt1 = plt.plot(over[:,0],over[:,1],'blue')
        else:
            plt1 = plt.plot(set0[:,0],set0[:,1],'green',label='Raw data')
            if over != None:
                plt1 = plt.plot(over[:,0],over[:,1],'blue')
        print("Kill plot to continue.")
        x1,x2,y1,y2 = plt.axis()
        if xmin != None:
            x1 = xmin
        if xmax != None:
            x2 = xmax
        if ymin != None:
            y1 = ymin
        if ymax != None:
            y2 = ymax
        plt.axis((x1,x2,y1,y2))
        plt.show()
    else:
       for i in range(set0.shape[0]):
           print("    %f     %f    "%(set0[i,0],set0[i,1]))


def getE0(set0):
    r"""
        (e0,ie0) = getE0(set0): returns the value and position of the maximum of derivative
    """
    tmp = numpy.gradient(set0[:,1])
    itmp = tmp.argmax()
    return set0[itmp,0],itmp

def getJump(set0):
    r"""
        getJump(set0): returns Jump. set0 must be in k-space. Jump is the ratio of the
                       ordinates of points with k in [1,2] over the average of ordinates 
                       of points with k<1
    """
    #average values k < -1 A^(-1)
    goodi = (set0[:,0]  < -1.0)
    y0 = (set0[goodi,1]).mean()
    #average values 1 < k < 2 A^(-1)
    goodi = (set0[:,0]  > 1.0) & (set0[:,0]  < 2.0)
    y1 = (set0[goodi,1]).mean()

    return numpy.abs(y1-y0)

def e2k(set0,e0=0.0):
    r"""
        e2k(set0,e0=0.0): converts from E (eV) to k (A^-1)
        note: we use the convention that points with E<e0 will have negative k
    """
    codata_ec = numpy.array(1.602176565e-19)
    codata_me = numpy.array(9.10938291e-31)
    codata_h = numpy.array(6.62606957e-34)
    codata_hbar = codata_h/2.0/numpy.pi
    #; converts a set in energy to a set in k
    #; the negative energies (below edge) are treated as negative k
    tmpx = set0[:,0] - e0
    ccte = numpy.sqrt(codata_ec*2*codata_me/codata_hbar/codata_hbar)*1e-10
    tmpxx = ((tmpx > 0) * 2-1) * numpy.sqrt(numpy.abs(tmpx)) * ccte
    set0[:,0] = tmpxx
    return set0

def k2e(set0,e0=0.0):
    r"""
        k2e(set0,e0=0.0): converts from k (A^-1) to E (eV)
    """
    codata_ec = numpy.array(1.602176565e-19)
    codata_me = numpy.array(9.10938291e-31)
    codata_h = numpy.array(6.62606957e-34)
    codata_hbar = codata_h/2.0/numpy.pi

    #; converts a set in k to energy
    #; the negative energies (below edge) are treated as negative k
    ccte = numpy.power(codata_hbar,2) / 2 / codata_me / codata_ec * 1e20
    tmpx = set0[:,0]
    tmpx = ((tmpx > 0) * 2-1) * tmpx * tmpx * ccte
    set1 = set0
    set1[:,0] = tmpx

    return set1

def polspl_evaluate(set2,xl,xh,c,nc,nr):
    r"""
        polspl_evaluate(set2,xl,xh,c,nc,nr): for internal use of postedge

     PURPOSE:
    	evaluate the combined spline fitted from its coefficients.
    
     INPUTS:
    	set2: the set with the original data
    	xl,xh arrays contain nr adjacent ranges over which to fit individual polynomials.  
           c array containing the polynomial coefficients resulting from the fit
           nc array that specifies how many poly coeffs to use in each range 
           nr the number of adjacent ranges
    	
     OUTPUTS:
    	a variable to receive a set with the same abscissas of the input one and 
        the coordinates evaluated from the fit parameters
    
     MODIFICATION HISTORY:
     	Written by:	Manuel Sanchez del Rio. ESRF,  February, 1993	
    	2009-05-13 srio@esrf.eu updated doc
        2014-12-04 srio@esrf.eu Translated to python
    """

    fit = set2*0.0
    #;change xl(1) and xh(nr) to extrapolate the fit
    xl[1] = numpy.min(set2[0,:])
    xh[nr] = numpy.max(set2[0,:])

    #;
    #; calculatest the first point
    #;
    xval=set2[0,0]
    yval=0.0
    for k in range(1,int(nc[1]+1)):
        print(k)
        yval =  yval+ c[k] * numpy.power(xval,(k-1))
    fit[0,0] = xval
    fit[1,0] = yval

    #;
    #; now the rest of the points
    #;
    for i in range(len(set2[0,:])):  # loop over all the points
        for j in range(1,int(nr+1)): # loop over the # of intervals
            if ((set2[0,i] > xl[j]) and (set2[0,i] <= xh[j])):
                cstart=numpy.sum(nc[0:j])
                xval = set2[0,i]
                yval = 0.0
                for k in range(1,int(nc[j]+1)): 
                    yval =  yval+ c[cstart+k] * numpy.power(xval,(k-1))
                fit[0,i] = xval
                fit[1,i] = yval
    return fit

def polspl(x,y,w,npts,xl,xh,nr,nc):
    r"""
        polspl(x,y,w,npts,xl,xh,nr,nc): for internal use of postedge

     PURPOSE:
    	polynomial spline least squares fit to data points Y(I).
    	only the function and it's first derivative are matched at the knots,
    	in order to give more degrees of freedom in the fit.
    
     INPUTS:
    	x(i),i=1,npts           abscissas
    	y(i),i=1,npts           ordinates
    	w(i),i=1,npts           weighting factor in least squares fit
    	fit minimizes the sum of w(i)*(y(i)-poly(x(i)))**2
    	if uniform weighting is desired, w(i) must be 1.
    	npts: points in x,y arrays.  xl,xh arrays contain NR adjacent ranges
    	over which to fit individual polynomials.  Array nc specifies
    	how many poly coeffs to use in each range.
    
     OUTPUTS:
    	array with all coeffs, the first nc(1) of which belong to the first range,
    	the second nc(2) of which belong to the second range, and so forth.
    
     SIDE EFFECTS:
    	Quite inefficient, because it uses a lot of loops inherited from
    	the Fortran code. However, for small set of data it is useful.
    
     PROCEDURE:
    	(Translated from a Fortran Code)
    	The method here is to fit ordinary polynomials in X, not B-splines,
    	order to save space on a mini-computer.  This means that the
    	is rather poorly conditioned, and hence the limits on the
    	of the polynomial.  The method of solution is Lagrange's
    	multipliers for the knot constraints and gaussian
    	to solve the linear system.
    
     MODIFICATION HISTORY:
     	Written by:	Manuel Sanchez del Rio. ESRF February, 1993	
        2014-12-04 srio@esrf.eu Translated to python
    
        this subroutine is a translation of the fortran subroutine
        poslpl.for (found in the Frascati's package of EXAFS data analysis)
    
    """

    # ;
    # ; few definitions
    # ;
    df = numpy.zeros(26)  
    a = numpy.zeros((36,37))  
    nbs = numpy.zeros(11,dtype=int)
    xk = numpy.zeros(10)  
    c = numpy.zeros(36)  
    j=0 
    i=0  
    ne_idl=0 
    n = 0 
    k = 0 
    ibl = 0
    ns = 0  
    ns1 = 0

    nbs[1]=1
    for i in range(1,nr+1):
        n=n+int(nc[i])
        nbs[i+1]=n+1
        if xl[i] < xh[i]: 
            pass
        else:
            t=xl[i]
            xl[i]=xh[i]
            xh[i]=t

    n=n+2*(nr-1)
    n1=n+1
    xl[nr+1]=0.
    xh[nr+1]=0.

    for ibl in range(1,nr+1):
        xk[ibl]=.5*(xh[ibl]+xl[ibl+1])
        if (xl[ibl] > xl[ibl+1]):
            xk[ibl]=.5*(xl[ibl]+xh[ibl+1])
        ns=nbs[ibl]
        ne_idl=nbs[ibl+1]-1
        for i in range(1,npts+1):
            if((x[i] < xl[ibl]) or (x[i] > xh[ibl])): 
                pass
            else:
                df[ns]=1.0
                ns1=ns+1
                for j in range(ns1,ne_idl+1):
                    df[j]=df[j-1]*x[i]
                for j in range(ns,ne_idl+1): 
                    for k in range(j,ne_idl+1): 
                        a[j,k]=a[j,k]+df[j]*df[k]*w[i]
                    a[j,n1]=a[j,n1]+df[j]*y[i]*w[i]

    ncol=nbs[nr+1]-1
    nk=nr-1

    if (nk == 0): 
        pass
    else:
        for ik in range(1,nk+1):
            ncol=ncol+1
            ns=nbs[ik]
            ne_idl=nbs[ik+1]-1
            a[ns,ncol]=-1.
            ns=ns+1
            for i in range(ns,ne_idl+1):
                a[i,ncol]=a[i-1,ncol]*xk[ik]
            ncol=ncol+1
            a[ns,ncol]=-1.
            ns=ns+1
            if (ns > ne_idl): 
                pass
            else:
                for i in range(ns,ne_idl+1):
                    a[i,ncol]=(ns-i-2)*numpy.power(xk[ik],(i-ns+1))
            ncol=ncol-1
            ns=nbs[ik+1]
            ne_idl=nbs[ik+2]-1
            a[ns,ncol]=1.0
            ns=ns+1
            for i in range(ns,ne_idl+1):
                a[i,ncol]=a[i-1,ncol]*xk[ik]
            ncol=ncol+1
            a[ns,ncol]=1.0
            ns=ns+1
            if (ns > ne_idl): 
                pass
            else:
                for i in range(ns,ne_idl+1): 
                    a[i,ncol]=(i-ns+2)*numpy.power(xk[ik],(i-ns+1))

    for i in range(1,n+1):
        i1=i-1
        for j in range(1,i1+1): 
            a[i,j]=a[j,i]
    nm1=n-1

    for i in range(1,nm1+1): 
        i1=i+1
        m=i
        t=numpy.abs(a[i,i])
        for j in range(i1,n+1): 
            if (t >= numpy.abs(a[j,i])):
                pass
            else:
                m=j
                t=numpy.abs(a[j,i])
        if (m == i): 
            pass
        else:
            for j in range(1,n1+1): 
                t=a[i,j]
                a[i,j]=a[m,j]
                a[m,j]=t
        for j in range(i1,n+1): 
            t=a[j,i]/a[i,i]
            for k in range(i1,n1+1): 
                a[j,k]=a[j,k]-t*a[i,k]
    c[n]=a[n,n1]/a[n,n]
    for i in range(1,nm1+1): 
        ni=n-i
        t=a[ni,n1]
        ni1=ni+1
        for j in range(ni1,n+1): 
            t=t-c[j]*a[ni,j]
        c[ni]=t/a[ni,ni]

    return c

def polspl_test():
    r"""
        polspl_test(): to test polspl ()
    """
    set22 = numpy.loadtxt('set22.dat')
    set22 = set22.T
    
    npts = len(set22[1,:])
    w = numpy.ones(npts+1)
    xx = numpy.zeros(npts+1)
    yy = numpy.zeros(npts+1)
    #w=w*0.0+1.0

    xx[1:npts+1]=set22[0,:]  
    yy[1:npts+1]=set22[1,:]
    xl = numpy.array( [ 0.0000000, 0.0000000, \
                        7.6354497, 15.270899, 0.0000000,\
                        0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000 ])
    xh = numpy.array( [  0.0000000, 7.6354497,\
                         15.270899, 22.906349, 0.0000000, 0.0000000,\
                         0.0000000, 0.0000000, 0.0000000, 0.0000000 ] )
    nc = numpy.array( [ 0.0000000, 4.0000000,\
                        4.0000000, 4.0000000, 0.0000000, 0.0000000,\
                        0.0000000, 0.0000000, 0.0000000, 0.0000000 ] )
    nr =       3
    c = polspl(xx,yy,w,npts,xl,xh,nr,nc)
    print("set22.shape",set22.shape)
    fit = polspl_evaluate(set22,xl,xh,c,nc,nr)
    print("fit.shape",fit.shape)
    print("c: ",c)
    print("fit: ",fit)
    return


def postEdge(set2,kmin=None,kmax=None,polDegree=[3,3,3],knots=None):
    r"""
        postEdge(set2,xrange=None,polDegree=[3,3,3],knots=None):
    
     PURPOSE:
    	This procedure calculates the post edge fit of a xafs spectrum
    
     INPUTS:
    	set2: input set of data
    
     KEYWORD PARAMETERS:
        kmin the bottom limit for the fit (defaults kmin=0)
        kmax the upper limit for the fit (defaults max)
    
     OUTPUTS:
    	a set with the fit
    
     MODIFICATION HISTORY:
     	Written by:	Manuel Sanchez del Rio. ESRF
    	February, 1993	
        1996-08-13 MSR (srio@esrf.fr) changes wmenu->wmenu2 and
                   xtext->widget_message
    	1998-10-01 srio@esrf.fr adapts for delia.
    	2000-02-12 MSR (srio@esrf.fr) adds Dialog_Parent keyword
    	2014-12-04 srio@esrf.eu Translated to python

    """
    #Note that in/out arrays are numpy way: numpy.array((npoints,2))

    xl = numpy.zeros(10)
    xh = numpy.zeros(10)
    c = numpy.zeros(36)
    nc = numpy.zeros(10)
    if len(polDegree) > 10: 
        print("Error: Maximum number of intervals is 10")
        print("       Number of intervals forced to 10")
        polDegree = polDegree[0:9]

    x1 = 0.0 # set2[:,0].min()
    x2 = set2[:,0].max()

    if kmin != None:
        x1 = kmin
    if kmax != None:
        x2 = kmax

    xrange = [x1,x2]
    print("++++++++++++++++++",xrange)

    if (knots != None):
        if ( (len(polDegree)+1) != len(knots) ):
            print("Error: dimension of knots must be dimension of polDegree+1")
            print("       Forced automatic (equidistant) knot definition.")
            knots = None
        else: 
            xrange = knots[0,-1]


    nr = len(polDegree)
    xl[1] = xrange[0]
    xh[nr] = xrange[1]

    for i in range(1,nr+1):
        nc[i] = polDegree[i-1] + 1  

    if knots == None:
        step = (xh[nr]-xl[1])/float(nr)
        for i in range(1,nr):
            xl[i+1] = xl[i] + step
            xh[i]   = xl[i+1]
    else:
        for i in range(1,nr):
            xl[i+1] = knots[i-1]
            xh[i]   = xl[i+1]

    #
    # select only points in selected interval
    #
    goodi = (set2[:,0] >= xrange[0]) & (set2[:,0] <= xrange[1])
    set22 = set2[goodi,:]

    print(' Number of fitting points: %d'%(len(set22[:,0])))
    print(' polynomials used for fitting: %d'%(nr))
    print('#        degree   min      max')
    for i in range(1,nr+1):
        print("%d %9d %9.2f %9.2f "%(i,nc[i]-1,xl[i],xh[i]))

    # ;
    # ; call spline
    # ;
    npts = len(set22[:,0])
    w = numpy.ones(npts+1)
    xx = numpy.zeros(npts+1)
    yy = numpy.zeros(npts+1)
    xx[1:] = set22[:,0]
    yy[1:] = set22[:,1]

    c = polspl(xx,yy,w,npts,xl,xh,nr,nc)
    print("c:",c)

    #TODO: polspl_evaluate receives and returns arrays like IDL (2,npoints)
    fit0 = polspl_evaluate(set2.T,xl,xh,c,nc,nr)

    return fit0.T

def window_ftr(setin,window=1,windpar=0.2,wrange=None):

    r"""
        window_ftr(setin,window=1,windpar=0.2,wrange=None)
     
      PURPOSE:
     	This procedure calculates and applies a weighting window to a set
     
      INPUTS:
     	setin:	either:
                 numpy.array(npoints,ncols) set of data  (CASE A)
                 numpy.array(npoints) array of abscissas  (CASE B)
      
      OUTPUT:
     	depends on the case: 
           CASE A: numpy.array(npoints,ncol) set with the weigted set (in index [:,1])
           CASE B: numpy.array(npoints) the values of the weights
     
      KEYWORD PARAMETERS:
     	window = kind of window:
     		1 Gaussian Window (default)
     		2 Hanning Window
     		3 Box
     		4 Parzen (triangular)
     		5 Welch
     		6 Hamming
     		7 Tukey
     		8 Papul
     	windpar Parameter for windowing
     		If WINDOW=(2,3,4,5,6) this sets the width of the appodization (default=0.2)
     	wrange = [xmin,xmax] the limits of the window. If wrange
     		is not set, the take the minimum and maximum values
     		of the abscisas. The window has value zero outside
     		this interval.
     
      MODIFICATION HISTORY:
      	Written by:	Manuel Sanchez del Rio. ESRF
     	March, 1993
     	96-08-14 MSR (srio@esrf.fr) adds names keyword.
     	06-03-14 srio@esrf.fr always exits "names"
     	2014-12-03 srio@esrf.eu translated to python
    ;-
    """
    names = ['1 Gaussian', '2 Hanning','3 Box','4 Parzen','5 Welch','6 Hamming','7 Tukey','8 Papul','9 Kaiser']

    print("Using window ",names[window-1])

    si = setin.shape
    if len(si) >=2:  # input set
        tk = setin[:,0]
    else:            # input array
        tk = setin

    if wrange == None:
        xmax = tk.max()
        xmin = tk.min()
    else:
        xmin = wrange[0]
        xmax = wrange[1]

    xp = (xmax + xmin) / 2.
    xm = xmax - xmin
    apo1 = xmin + windpar
    apo2 = xmax - windpar

    npoint = len(tk)
    wind = numpy.ones(npoint)

    if window == 1: # Gaussian
        wind = numpy.power(( (tk - xp) /xm),2)
        wind = numpy.exp(-wind * 9.2)

    if window == 2: # Hanning
        for i in range(npoint):
            if tk[i] <= apo1:
                wind[i] = 0.5*(1.0-numpy.cos(numpy.pi*(tk[i]-xmin)/windpar))
            if tk[i] >= apo2:
                wind[i] = 0.5*(1.0+numpy.cos(numpy.pi*(tk[i]-apo2)/windpar))

    if window == 3: # Box
        for i in range(npoint):
            if tk[i] <= apo1:
                wind[i] = 0.0
            if tk[i] >= apo2:
                wind[i] = 0.0

    if window == 4: # Parzen (triangle)
        for i in range(npoint):
            if tk[i] <= apo1:
                wind[i] = (tk[i]-xmin)/windpar
            if tk[i] >= apo2:
                wind[i] = 1 - (tk[i]-apo2)/windpar

    if window == 5: # Welch
        for i in range(npoint):
            if tk[i] <= apo1:
                wind[i] = 1.0 - numpy.power( ( (tk[i]-apo1) / windpar), 2)
            if tk[i] >= apo2:
                wind[i] =  1.0 - numpy.power( (tk[i]-apo2) / windpar, 2 )

    if window == 6: # Hamming
        for i in range(npoint):
            if tk[i] <= apo1:
                wind[i] = 1.08 - (.54+0.46*numpy.cos(numpy.pi*(tk[i]-xmin)/windpar))
            if tk[i] >= apo2:
                wind[i] =   1.08 - (.54-0.46*numpy.cos(numpy.pi*(tk[i]-apo2)/windpar))

    if window == 7: # Tukey
        for i in range(npoint):
            if tk[i] <= apo1:
                wind[i] = 1.0 - numpy.power(numpy.cos(0.5*numpy.pi*(tk[i]-xmin)/windpar),2)
            if tk[i] >= apo2:
                wind[i] = numpy.power(numpy.cos(-0.5*numpy.pi*(tk[i]-apo2)/windpar),2)

    if window == 8: # Papul
        for i in range(npoint):
            if tk[i] <= apo1:
                a=(1./numpy.pi)*numpy.sin(numpy.pi*(tk[i]-xmin)/windpar) + \
                  (1.-(tk[i]-xmin)/windpar)*numpy.cos(numpy.pi*(tk[i]-xmin)/windpar)
                wind[i] = 1.0 - a
            if tk[i] >= apo2:
                a=(1./numpy.pi)*numpy.sin(numpy.pi*(tk[i]-apo2)/windpar) + \
                  (1.-(tk[i]-apo2)/windpar)*numpy.cos(numpy.pi*(tk[i]-apo2)/windpar)
                wind[i] = a 

    # not implemented as require special functions and dependency on scipy
    #   9: begin                    ; kasel
    #        wind=beseli( windpar*sqrt(1.-((tk-xp)/xm*2.)^2),0 )/  $
    #        beseli(windpar,0)
    #      end

    if len(si) >=2:  # output weighted set
        setout = numpy.zeros((npoint,2))
        setout[:,0] = tk
        setout[:,1] = wind*setin[:,1]
    else:            # output window array
        setout = wind

    return setout

def fastftr(ftrin,npoint=4096,rrange=[0.,6.],kstep=0.04):
    r"""
        fastftr(ftrin,npoint=4096,rrange=[0.,6.],kstep=0.04):
    
     PURPOSE:
    	This procedure calculates the Fast Fourier Transform of a set

     INPUTS:
    	ftrin:  a 2 or 3 col set with k,real(chi),imaginary(chi)
    
     OUTPUTS:
    	This function returns a 4-columns array (ftrout) with
    	the congugare variable (R) in column 0, the modulus of the
    	FT in col 1, the real part in col 2 and the imaginary part in
    	col 3.

     KEYWORD PARAMETERS:
    	rrange=[rmin,rmax] : range of the congugated variable for 
    		the transformation (default = [0.,6.])
    	npoint= number of points of the the fft calculation (default = 4096)
    	kstep = step of the k variable for the interpolation (default=0.04)
    
     MODIFICATION HISTORY:
     	Written by:	Manuel Sanchez del Rio. ESRF, March, 1993	
        20141204 srio@esrf.eu Translated to python
    """

    npoint2 = len(ftrin[:,0])
    xmin = ftrin[0,0]
    xmax = ftrin[-1,0]
    b = numpy.zeros( (npoint,2) )


    # ;
    # ; creates the b set with the interpolated values of ftrin
    # ;
    b[:,0] = numpy.linspace(0.0,npoint-1,npoint) * kstep
    b[:,1] = numpy.interp( b[:,0] , ftrin[:,0], ftrin[:,1], left=0.0, right=0.0)


    # ; calculates the fft and generates the congugated variable (rr)

    ff = numpy.fft.ifft(b[:,1])
    rstep = numpy.pi / npoint / kstep
    rr = numpy.linspace(0.0,npoint-1,npoint) * rstep


    # ;
    # ; prepare the results
    # ;

    coef = npoint * kstep / numpy.sqrt(numpy.pi) * numpy.sqrt(2.)
    f12 = coef*numpy.real(ff)             # real part of fft
    f13 = coef*numpy.imag(ff)*(-1.)       # imaginary part of fft

    # ;
    # ; cut the results to the selected interval in r (rrange)
    # ;

    goodi = (rr  >= rrange[0]) & (rr  <= rrange[1])
    f13 = f13[goodi]
    f12 = f12[goodi]
    f10 = rr[goodi]
    f11 = numpy.sqrt( f12*f12 + f13*f13)

    # ;
    # ; define the result array
    # ;
    fourier = numpy.zeros((len(f10),4))
    fourier[:,0] = f10
    fourier[:,1] = f11
    fourier[:,2] = f12
    fourier[:,3] = f13

    return fourier

def fastbftr(fourier,npoint=4096,krange=[2.0,12.0],rstep=None,rmin=None,rmax=None):
    r"""
        fastbftr(fourier,npoint=4096,krange=[2.0,12.0],rstep=None,rmin=None,rmax=None)

     PURPOSE:
    	This procedure calculates the Back Fast Fourier Transform of a set
    
     INPUTS:
           fourier:  a 4 col set with r,modulus,real and imaginary part
                   of a Fourier Transform of an Exafs spectum, as produced
                   by FTR or FASTFTR procedures
    	
     KEYWORD PARAMETERS:
    	krange=[kmin,kmax] : range of the conjugated variable for 
    		the transformation (default = [2,15])
    	npoint= number of points of the the fft calculation (default = 4096)
    	rstep = when this keyword is set then the fourier set is 
    		interpolated using the indicated value as step. Otherwise
    		the fourier set is not interpolated.
        rmin = the mimimun r for the back fourier filtering
        rmax = the maximum r for the back fourier filtering

     OUTPUTS:
           This procedure returns a 4-columns set (backftr) with
           the conjugated variable (k) in column 0, the real part of the
           BFT in col 1, the modulus in col 2 and the phase in col 3.
    
     MODIFICATION HISTORY:
     	Written by:	Manuel Sanchez del Rio. ESRF March, 1993	
    	98-10-26 srio@esrf.fr uses Dialog_Message for error messages.
        20141204 srio@esrf.eu Translated to python
    """

    kmin = krange[0]
    kmax = krange[1]

    npt = len(fourier[:,0])
    fou = numpy.zeros((npoint,4))

    if rmin == None:
        rmin = (fourier[:,0]).min()
    if rmax == None:
        rmax = (fourier[:,0]).max()

    #;
    #; fill "fou" set
    #;
    if rstep == None: #;--- no interpolation
        nn = int(npt/2)
        rstep = fourier[nn+1,0] - fourier[nn,0]
        rstep2 = fourier[nn+2,0] - fourier[nn+1,0]
        rdiff = numpy.abs (rstep - rstep2)
        print(' back rstep = %f'%(rstep))
        print(' rdiff = %f'%rdiff)
        if (rdiff >= 1e-6):
            print('r griding is not regular; Use rstep keyword -> Abort')
            return fou
 
        ptstart = int(rmin/rstep)
        print(' ptstart = %d'%ptstart)
        print(' ptstart+npt = %d'%(ptstart+npt))
        fou[ptstart:ptstart+npt,:]=fourier
    else: #;--- interpolation
        fou[:,0] = numpy.linspace(0,0,npoint-1,npoint)*rstep
        fou[:,1] = numpy.interp(fou[:,0],fourier[:,0],fourier[:,1],left=0.0,right=0.0)
        fou[:,2] = numpy.interp(fou[:,0],fourier[:,0],fourier[:,2],left=0.0,right=0.0)
        fou[:,3] = numpy.interp(fou[:,0],fourier[:,0],fourier[:,3],left=0.0,right=0.0)

    #;
    #; call back fft
    #;
    c = fou[:,2] - 1.0j * fou[:,3]
    af = numpy.fft.fft(c)
    
    #;
    #; create the array of the conjugated variable
    #;
    kstep = numpy.pi/npoint/rstep
    kk = numpy.linspace(0.0,npoint-1,npoint)*kstep


    #;
    #; prepare the output array
    #;
    coef = npoint*kstep/numpy.sqrt(numpy.pi)*numpy.sqrt(2.) # coefficienu used for direct fft
    coef1 = 2./coef                                         # 2 because we are only 
    afr = coef1 * af.real                                   # real part of back fft
    afi = coef1 * af.imag                                   # imaginary part of back fft

    #;
    #; cut the results to the selected interval in k (krange)
    #;

    goodi = (kk  >= kmin) & (kk  <= kmax)
    afr = afr[goodi]
    afi = afi[goodi]
    afk = kk[goodi]
    nptout = len(afr)

    #;
    #; define the output set
    #;
    backftr = numpy.zeros((nptout,4))
    backftr[:,0] = afk                  # the conjugated variable (k [A^-1])
    backftr[:,1] = afr                  # the real part of backftr or atra
    backftr[:,2] = numpy.sqrt(afr*afr+afi*afi)    # the modulus of backftr
    backftr[:,3] = numpy.arctan2(afi,afr)          # the phase

    return backftr


if __name__ == '__main__':

    #
    # plotting setup
    #
    pltOk = True
    try:
        import matplotlib.pylab as plt
    except ImportError:
        pltOk = False
        print("failed to import matplotlib. No on-line plots.")

    #
    #load mu(E); E in eV
    fileIn = "Ge_calib.dat"
    set0 = numpy.loadtxt(fileIn)
    plotSet(set0,xtitle="photon Energy [eV]",ytitle="$\mu$ [a.u.]", \
        toptitle="Raw data from: "+fileIn)

    #;
    #; Edge Value ---------------------------------------------------
    #;
    print('********************* Edge value ****************************')
    e0,ie0 = getE0(set0)
    print(' The selected Eo from the maximum of the derivative is %f eV'%(e0))

    set0[:,0] -= e0
    #plotSet(set0,xtitle="E-Eo [eV]",ytitle="$\mu$ [a.u.]", \
    #   toptitle="Raw data from: "+fileIn)

    #;
    #; Pre edge ----------------------------------------------------
    #;
    print('**************************  Pre edge  *********************')
    ieFrom = 0
    ieTo = ie0 - int(ie0*0.9)

    # substract pre-edge linear fit
    p = numpy.polyfit(set0[ieFrom:ieTo,0], set0[ieFrom:ieTo,1], 1)

    setFit = numpy.copy(set0)
    setFit[:,1] = (p[0] * setFit[:,0] + p[1])

    plotSet(set0,over=setFit,xtitle="E-Eo [eV]",ytitle="$\mu$ [a.u.] ", \
       toptitle="Pre-edge fit")

    #set0[:,1] -= (p[0] * set0[:,0] + p[1])
    set0[:,1] -= setFit[:,1]

    # change abscissas to wavenumber
    set0 = e2k(set0)

    #plotSet(set0,xtitle="k [$A^{-1}$]",ytitle="$\mu - \mu_{pre}$ [a.u.] ", \
    #   toptitle="Raw data from: "+fileIn)

    #;
    #; Post edge ----------------------------------------------------
    #;
    print('**************************  Post edge  *********************')

    #write file for polspl_test()
    #numpy.savetxt("set22.dat",set0)
    #print("File written to disk: set22.dat")

    fit0 = postEdge(set0,polDegree=[3,2,2,2],kmin=2.)

    plotSet(set0,fit0,xtitle="k [$A^{-1}$]",ytitle=" fit ",toptitle="post edge",xmin=2,ymin=-.5,ymax=1.5)

    #;
    #; Normalization ----------------------------------------------------
    #;
    print('**************************  Normalization  *********************')

    i_menu_nor = 2  #   1=experimental, 2=constant, 3=Lengeler-Eisenberg

    #use E scale (mandatory)

    set1 = k2e(set0)
    fit1 = k2e(fit0)

    # get jump
    if ( (i_menu_nor == 2) or (i_menu_nor == 3)):
        jump = getJump(set1)
        print("Got jump value = %f"%jump)

    if i_menu_nor == 3:
        if e0 <= 0.01:
            print("Error applying Lengeler-Eisenberg normalization:")
            print("  E0 must be defined and not zero (E0=%f eV)"%e0)
            print("  Forcing to 'constant' normalization")
            i_menu_nor = 2
        else:
            set2 = set1
            #set2[:,0] += e0
            set2[:,1] = ( set1[:,1]  - fit1[:,1] ) / jump / \
                        (1. - (8./3.)*(set1[:,0])/e0)


    if i_menu_nor == 2:
        set2 = set1
        set2[:,1] = ( set1[:,1]  - fit1[:,1] ) / jump


    if i_menu_nor == 1:
        set2 = set1
        set2[:,1] = ( set1[:,1]  - fit1[:,1] ) /  fit1[:,1]



    # back to k
    set2 = e2k(set2)

    #remove points with k<2 
    goodi = (set2[:,0] > 2.0)
    set2 = set2[goodi,:]

    plotSet(set2,xtitle="k [$A^{-1}$]",ytitle="$\chi$", toptitle="EXAFS")

    #;
    #; Fourier transform ----------------------------------------------------
    #;
    print('**************************  Fourier transform  *********************')


    #numpy.savetxt("setMu.dat",set2)
    #print("File setMu.dat written to file.")

    #window
    set2 = window_ftr(set2,window=8,windpar=3)
    plotSet(set2,xtitle="k [$A^{-1}$]",ytitle="$\chi$", toptitle="WINDOWED EXAFS")

    #FT
    setFT = fastftr(set2,npoint=4096,rrange=[0.,7.],kstep=0.02)
    plotSet(setFT,xtitle="R [$A$]",ytitle=" |FT| ", toptitle="EXAFS FT")

    #;
    #; Fourier filter ----------------------------------------------------
    #;
    print('**************************  Fourier Filter  *********************')

    #BACK FT
    setBFT = fastbftr(setFT,rmin=1.0,rmax=3.0,krange=[2.0,20.0])

    plotSet(setBFT,xtitle="K [$A^{-1}$]",ytitle="$\chi$", toptitle="EXAFS BFT FILTERED in (1,3)")



