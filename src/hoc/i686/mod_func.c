#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," IF_BG5.mod");
    fprintf(stderr," alphaisyn.mod");
    fprintf(stderr," alphasyn.mod");
    fprintf(stderr," expisyn.mod");
    fprintf(stderr," refrac.mod");
    fprintf(stderr," reset.mod");
    fprintf(stderr," stdwa_softlimits.mod");
    fprintf(stderr," stdwa_songabbott.mod");
    fprintf(stderr," stdwa_symm.mod");
    fprintf(stderr," tmgsyn.mod");
    fprintf(stderr," vecstim.mod");
    fprintf(stderr, "\n");
  }
  _IF_BG5_reg();
  _alphaisyn_reg();
  _alphasyn_reg();
  _expisyn_reg();
  _refrac_reg();
  _reset_reg();
  _stdwa_softlimits_reg();
  _stdwa_songabbott_reg();
  _stdwa_symm_reg();
  _tmgsyn_reg();
  _vecstim_reg();
}
