/* Created by Language version: 6.2.0 */
/* NOT VECTORIZED */
#include <stdio.h>
#include <math.h>
#include "scoplib.h"
#undef PI
 
#include "md1redef.h"
#include "section.h"
#include "md2redef.h"

#if METHOD3
extern int _method3;
#endif

#undef exp
#define exp hoc_Exp
extern double hoc_Exp();
 
#define _threadargscomma_ /**/
#define _threadargs_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define ping _p[0]
#define index _p[1]
#define etime _p[2]
#define space _p[3]
#define _tsav _p[4]
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_element();
 static double _hoc_play();
 static int _mechtype;
extern int nrn_get_mechtype();
 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(_ho) Object* _ho; { void* create_point_process();
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt();
 static double _hoc_loc_pnt(_vptr) void* _vptr; {double loc_point_process();
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(_vptr) void* _vptr; {double has_loc_point();
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(_vptr)void* _vptr; {
 double get_loc_point_process(); return (get_loc_point_process(_vptr));
}
 static _hoc_setdata(_vptr) void* _vptr; { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 /* connect user functions to hoc names */
 static IntFunc hoc_intfunc[] = {
 0,0
};
 static struct Member_func {
	char* _name; double (*_member)();} _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "element", _hoc_element,
 "play", _hoc_play,
 0, 0
};
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "ping", "ms",
 0,0
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(), nrn_init(), nrn_state();
 static void _hoc_destroy_pnt(_vptr) void* _vptr; {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static char *_mechanism[] = {
 "6.2.0",
"VecStim",
 "ping",
 0,
 0,
 0,
 0};
 
static void nrn_alloc(_prop)
	Prop *_prop;
{
	Prop *prop_ion, *need_memb();
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 5, _prop);
 	/*initialize range parameters*/
 	ping = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 5;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static _initlists();
 
#define _tqitem &(_ppvar[2]._pvoid)
 static _net_receive();
 typedef (*_Pfrv)();
 extern _Pfrv* pnt_receive;
 extern short* pnt_receive_size;
 _vecstim_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,0, 0, 0, nrn_init,
	 hoc_nrnpointerindex,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func,
	 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
  hoc_register_dparam_size(_mechtype, 3);
 add_nrn_artcell(_mechtype, 2);
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 VecStim /home/andrew/dev/pyNN_trunk/src/hoc/i686/vecstim.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static _modl_cleanup(){ _match_recurse=1;}
static element();
static play();
 
static _net_receive (_pnt, _args, _lflag) Point_process* _pnt; double* _args; double _lflag; 
{    _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 1.0 ) {
     net_event ( _pnt, t ) ;
     element ( _threadargs_ ) ;
     if ( index > 0.0 ) {
       if ( etime < t ) {
         printf ( "Warning in VecStim: spike time (%g ms) before current time (%g ms)\n" , etime , t ) ;
         }
       else {
         artcell_net_send ( _tqitem, _args, _pnt, t +  etime - t , 1.0 ) ;
         }
       }
     }
   else if ( _lflag  == 2.0 ) {
     if ( index  == - 2.0 ) {
       index = 0.0 ;
       while ( etime < t  && index >= 0.0 ) {
         element ( _threadargs_ ) ;
         }
       if ( index > 0.0 ) {
         artcell_net_send ( _tqitem, _args, _pnt, t +  etime - t , 1.0 ) ;
         }
       }
     artcell_net_send ( _tqitem, _args, _pnt, t +  ping , 2.0 ) ;
     }
   } }
 
/*VERBATIM*/
extern double* vector_vec();
extern int vector_capacity();
extern void* vector_arg();
 
static int  element (  )  {
   
/*VERBATIM*/
    { void* vv; int i, size; double* px;
        i = (int)index;
        if (i >= 0) {
            vv = *((void**)(&space));
            if (vv) {
                size = vector_capacity(vv);
                px = vector_vec(vv);
                if (i < size) {
                    etime = px[i];
                    index += 1.;
                } else {
                    index = -1.;
                }
            } else {
                index = -1.;
            }
        }
    }
  return 0; }
 
static double _hoc_element(_vptr) void* _vptr; {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 element (  ) ;
 return(_r);
}
 
static int  play (  )  {
   
/*VERBATIM*/
    void** vv;
    vv = (void**)(&space);
    *vv = (void*)0;
    if (ifarg(1)) {
        *vv = vector_arg(1);
    }
    index = -2;
  return 0; }
 
static double _hoc_play(_vptr) void* _vptr; {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 play (  ) ;
 return(_r);
}

static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   index = 0.0 ;
   element ( _threadargs_ ) ;
   if ( index > 0.0 ) {
     artcell_net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  etime - t , 1.0 ) ;
     }
   if ( ping > 0.0 ) {
     artcell_net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  ping + 1e-9 , 2.0 ) ;
     }
   }

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
 double _break, _save;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
 _break = t + .5*dt; _save = t;
 v=_v;
{
}}

}

static terminal(){}

static _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}
