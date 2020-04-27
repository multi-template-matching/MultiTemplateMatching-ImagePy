from imagepy import IPy
from imagepy.core.engine  import Free
from imagepy.core.manager import ImageManager
import cv2
import numpy as np

methods = {"Square-diff": cv2.TM_SQDIFF_NORMED,
           "cross-corr":  cv2.TM_CCORR_NORMED,
           "0-mean cross-corr": cv2.TM_CCOEFF_NORMED}

class Plugin(Free):
    title = 'Multi-Template-Matching'
    
    para = {'template':None,
            'image':None,
            'method':list(methods.keys())[-1], 
            'Nobject':1,
            'score-thresh':0.5,
            'overlap-thresh':0.4} # parameters name with default values
    
    
    view = [
            ("img", "template", "Template image", ""),
            ("img", "image", "Target image", ""),
            (int, 'Nobject', (1, np.inf), 0, 'Expected number of objects', ''),
            (list, "method", list(methods.keys()), str, 'Method for computation of score map', '(normalised)'),
            ('slide', 'score-thresh', (0, 1), 2, 'Score-threshold'),
            ('slide', 'overlap-thresh', (0, 1), 2, 'Overlap-threshold')
            ]

    def run(self, para=None):
        template = ImageManager.get(para["template"]).img # 2D array
        image    = ImageManager.get(para["image"]).img    # use imgs for a stack
        
        method = methods[ para['method'] ] # para['method'] returns the selected string
        #IPy.alert('Name:\t%s\r\nAge:\t%d'%(para['name'], para['age']))
        corrMap = cv2.matchTemplate(image, template, method) 
        IPy.show_img([corrMap], "Correlation map")