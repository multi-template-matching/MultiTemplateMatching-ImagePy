from imagepy.core.engine import Free
import cv2
import numpy as np

methods = {"Square-diff": cv2.TM_SQDIFF_NORMED,
           "cross-corr":  cv2.TM_CCORR_NORMED,
           "0-mean cross-corr": cv2.TM_CCOEFF_NORMED}

class Plugin(Free):
    title = 'Multi-Template-Matching'
    para = {'method':methods.keys()[-1], 
            'Nobject':1,
            'score-thresh':0.5,
            'overlap-thresh':0.4} # parameters name with default values
    
    
    view = [
            (int, 'Nobject', (1,np.inf), 0, 'Expected number of objects', ''),
            (list, "method", methods.keys(), str, 'Method for computation of score map', '(normalised)')
            ('slide', 'score-thresh', (0, 1), 2, 'Score-threshold'),
            ('slide', 'overlap-thresh', (0, 1), 2, 'Overlap-threshold')
            ]

    def run(self, para=None):
        method = methods[ para['method'] ] # para['method'] returns the selected string
        #IPy.alert('Name:\t%s\r\nAge:\t%d'%(para['name'], para['age']))