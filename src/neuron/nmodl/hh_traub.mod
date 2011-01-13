COMMENT
Modified Hodgkin-Huxley model
ENDCOMMENT
 
UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(uS) = (microsiemens)
        (S) = (siemens)
}
 
NEURON {
        SUFFIX hh_traub
        USEION na READ ena WRITE ina
        USEION k READ ek WRITE ik
        NONSPECIFIC_CURRENT il
        RANGE gnabar, gkbar, gl, el, gna, gk, vT
        GLOBAL minf, hinf, ninf, mtau, htau, ntau
	THREADSAFE : assigned GLOBALs will be per thread
}
 
PARAMETER {
        gnabar = 0.02 (S/cm2)	<0,1e9>
        gkbar = 0.006 (S/cm2)	<0,1e9>
        gl = 0.00001 (S/cm2)	<0,1e9>
        el = -60.0 (mV)
        vT = -63.0 (mV)
}
 
STATE {
        m h n
}
 
ASSIGNED {
        v (mV)
        ena (mV)
        ek (mV)
        
	gna (S/cm2)
	gk (S/cm2)
        ina (mA/cm2)
        ik (mA/cm2)
        il (mA/cm2)
        minf hinf ninf
	mtau (ms) htau (ms) ntau (ms)
}
 
BREAKPOINT {
        SOLVE states METHOD cnexp
        gna = gnabar*m*m*m*h
	ina = gna*(v - ena)
        gk = gkbar*n*n*n*n
	ik = gk*(v - ek)      
        il = gl*(v - el)
}
 
 
INITIAL {
        : the following (commented out) is the preferred initialization
	:rates(v)
	:m = minf
	:h = hinf
	:n = ninf
        : but for compatibility with NEST, we use the following
        m = 0
        h = 0
        n = 0
}

DERIVATIVE states {  
        rates(v)
        m' =  (minf-m)/mtau
        h' = (hinf-h)/htau
        n' = (ninf-n)/ntau
}
 
PROCEDURE rates(v(mV)) {
        LOCAL  alpha, beta, sum, u
        TABLE minf, mtau, hinf, htau, ninf, ntau FROM -100 TO 100 WITH 200

UNITSOFF
        u = v - vT
                :"m" sodium activation system
        alpha = 0.32 * vtrap(13-u, 4)
        beta =  0.28 * vtrap(u-40, 5)
        sum = alpha + beta
	mtau = 1/sum
        minf = alpha/sum
                :"h" sodium inactivation system
        alpha = 0.128 * exp((17-u)/18)
        beta = 4 / (exp((40-u)/5) + 1)
        sum = alpha + beta
	htau = 1/sum
        hinf = alpha/sum
                :"n" potassium activation system
        alpha = 0.032*vtrap(15-u, 5) 
        beta = 0.5*exp((10-u)/40)
	sum = alpha + beta
        ntau = 1/sum
        ninf = alpha/sum
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON
