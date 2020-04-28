from imagepy import IPy
from imagepy.core.engine  import Free
from imagepy.core.manager import ImageManager, RoiManager
#from imagepy.core.roi.rectangleroi import RectangleRoi # cant import roi
import cv2, MTM
import numpy as np

methods = {"Square-diff": cv2.TM_SQDIFF_NORMED,
           "cross-corr":  cv2.TM_CCORR_NORMED,
           "0-mean cross-corr": cv2.TM_CCOEFF_NORMED}

class ComputeMap(Free):
    title = 'Compute correlation map'
    
    para = {'template':None,
            'image':None,
            'method':list(methods.keys())[-1]
            }
    
    
    view = [
            ("img", "template", "Template image", ""),
            ("img", "image", "Target image", ""),
            (list, "method", list(methods.keys()), str, 'Method for computation of score map', '(normalised)'),
            ]

    def run(self, para=None):
        template = ImageManager.get(para["template"]).img # 2D array
        image    = ImageManager.get(para["image"]).img    # use imgs for a stack
        
        method = methods[ para['method'] ] # para['method'] returns the selected string
        
        corrMap = MTM.computeScoreMap(template, image, method)
        
        IPy.show_img([corrMap], "Correlation map")
        


class MultiTempMatching(Free):
    title = 'Multi-template-matching'
    
    para = {'template':None,
            'image':None,
            'method':list(methods.keys())[-1], 
            'Nobject':1,
            'score-thresh':0.5,
            'overlap-thresh':0.4
            } # parameters name with default values
    
    
    view = [
            ("img", "template", "Template image", ""),
            ("img", "image", "Target image", ""),
            (int, 'Nobject', (1, np.inf), 0, 'Expected number of objects', ''),
            (list, "method", list(methods.keys()), str, 'Method for computation of score map', '(normalised)'),
            ('slide', 'score-thresh', (0, 1), 2, 'Score-threshold'),
            ('slide', 'overlap-thresh', (0, 1), 2, 'Overlap-threshold')
            ]

    def run(self, para=None):
        templateTitle = para["template"]
        template = ImageManager.get(templateTitle).img # 2D array
        image    = ImageManager.get(para["image"]).img # use imgs for a stack
        
        method = methods[ para['method'] ] # para['method'] returns the selected string
        
        
        tableHit = MTM.matchTemplates([(templateTitle, template)], 
                                      image, 
                                      method,
                                      para["Nobject"],
                                      para["score-thresh"],
                                      para["overlap-thresh"]
                                      )
        
        # Display table
        IPy.show_table(tableHit, "Multi-Template-Matching detections")
        
        '''
        # Show ROI
        for index, row in tableHit.iterrows():
            x,y,width,height = row["BBox"]
            roi = RectangleRoi(x, y, x+width, y+height)
            name = "{}-{}".format(index, row["TemplateName"]) 
            RoiManager.add(name, roi)
        '''


# Define list of plugin defined here
plgs = [ComputeMap, MultiTempMatching]
