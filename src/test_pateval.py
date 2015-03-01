import PatternEvaluator as pe
import SamplingPattern as sp


mype = pe.PatternEvaluator()
mype.load_sens('../data/sens_6ch_128x128.npz')

# make sampling pat
mypat = sp.SamplingPattern()

# set up vectors for r= 2x2
mypat.yvec[::2,:]=0
mypat.xvec[:,::2]=0
#generate
mypat.calc_pattern()

#evaluate pattern

mype.eval_pattern(mypat)