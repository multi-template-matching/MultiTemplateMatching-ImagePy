from imagepy.core.engine import Free
import cv2

class Plugin(Free):
    title = 'Multi-Template-Matching'
    para = {'method':cv2.TM_CCOEFF_NORMED, 
            'Nobject':1,
            'score-thresh':0.5,
            'overlap-thresh':0.4} # parameters name with default values
    
    
    view = [
            (int, 'Nobject', (1,np.inf), 0, 'Expected number of objects', ''),
            ('slide', 'score-thresh', (0, 1), 2, 'Score-threshold'),
            ('slide', 'overlap-thresh', (0, 1), 2, 'Overlap-threshold')
            ] # Add method choice

    def run(self, para=None):
        IPy.alert('Name:\t%s\r\nAge:\t%d'%(para['name'], para['age']))