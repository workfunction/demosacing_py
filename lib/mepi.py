import numpy as np
import numba as nb
from numba import jitclass
from numba import int64
from numba import uint8
from numba import float64

spec = [
    ('width', int64),
    ('height', int64),
    ('outputWidth', int64),
    ('outputHeight', int64),
    ('inputImage', uint8[:, :]),
    ('scale', float64),
    ('mode', uint8)
]

@jitclass(spec)    
class MEPI:
    def __init__(self, oriImage, scale):
        self.width = oriImage.shape[1]
        self.height = oriImage.shape[0]

        self.scale = scale
        self.outputWidth = int(self.width * scale)
        self.outputHeight = int(self.height * scale)
        self.inputImage = np.copy(oriImage)
        
    def is_green(self, i, j):
        return (i%2) != (j%2)

    def weight_cal(self, vs, gamma=1.5):
        x_gamma = gamma/2
        
        tmp = (1-vs)
        vc3 = (tmp*vs*vs)*x_gamma
        vc0 = (tmp*tmp*vs)*x_gamma
        vc1 = tmp + 2*vc0 - 1*vc3
        vc2 = vs + 2*vc3 - 1*vc0
        vc0 = (-1)*vc0
        vc3 = (-1)*vc3

        return np.array([vc0, vc1, vc2, vc3])
    
    def weight_cal3(self, d):
        r = 4 + d * 4 - d * d
        vc1 = ((2 + d) * (2 - d))/r
        vc0 = (d * (2 + d))/r
        vc2 = (d * (2 - d))/r

        return np.array([vc0, vc1, vc2])

    def _DeltaH(self, i, j):
        if self.is_green(i, j):
            return (self.inputImage[i, j] / 2) + (self.inputImage[i, j-2] / 4) +   \
                (self.inputImage[i, j+2] / 4) - (self.inputImage[i, j-1] / 2) - \
                (self.inputImage[i, j+1] / 2)
        else:
            return (self.inputImage[i, j-1] / 2) + (self.inputImage[i, j+1] / 2) - \
                (self.inputImage[i, j-2] / 4) - (self.inputImage[i, j+2] / 4) - \
                (self.inputImage[i, j] / 2)

    def _DeltaV(self, i, j):
        if self.is_green(i, j):
            return (self.inputImage[i, j] / 2) + (self.inputImage[i-2, j] / 4) +   \
                (self.inputImage[i+2, j] / 4) - (self.inputImage[i-1, j] / 2) - \
                (self.inputImage[i+1, j] / 2)
        else:
            return (self.inputImage[i-1, j] / 2) + (self.inputImage[i+1, j] / 2) - \
                (self.inputImage[i-2, j] / 4) - (self.inputImage[i+2, j] / 4) - \
                (self.inputImage[i, j] / 2)

    def color_weight(self, weight, color):
        if color == False:
            return np.array([0, weight[0], 0, weight[1], 0, weight[2], 0, weight[3]])
        else:
            return np.array([weight[0], 0, weight[1], 0, weight[2], 0, weight[3], 0])
        
    def getDelta(self, dh, dv, gh, gv):
        s = gh + gv
        if s == 0:
            return dh
        return (gh * dv + gv * dh) / s
        
    def _getDelta(self, dh, dv, gh, gv):
        if gv <= gh:
            if gv * 4 <= gh:
                return dv
            elif gv * 2 <= gh:
                return (3 * dv + dh) / 4
            else:
                return (dv + dh) / 2
        else:
            if gh * 4 <= gv:
                return dh
            elif gh * 2 <= gv:
                return (3 * dh + dv) / 4
            else:
                return (dh + dv) / 2
    
    def _Delta(self, i, j , dy, dx):
        deltaH = np.zeros((8, 4))
        deltaV = np.zeros((8, 4))
        Delta = np.zeros((4, 4))

        for m in range(-3, 5, 1):
            for n in range(-1, 3, 1):
                deltaH[m+3, n+1] = self._DeltaH(i+m, j+n)
                deltaV[m+3, n+1] = self._DeltaV(i+n, j+m)

        Dh_C0 = np.dot(np.abs(deltaH[:, 0] - deltaH[:, 3]), np.array([0, 1/8, 0, 3/8, 0, 3/8, 0, 1/8]))
        Dh_C1 = np.dot(np.abs(deltaH[:, 0] - deltaH[:, 3]), np.array([1/8, 0, 3/8, 0, 3/8, 0, 1/8, 0]))
        
        if self.is_green(i, j):
            Dv_C0 = np.dot(np.abs(deltaV[:, 0] - deltaV[:, 3]), np.array([0, 1/8, 0, 3/8, 0, 3/8, 0, 1/8]))
            Dv_C1 = np.dot(np.abs(deltaV[:, 0] - deltaV[:, 3]), np.array([1/8, 0, 3/8, 0, 3/8, 0, 1/8, 0]))
        else:
            Dv_C1 = np.dot(np.abs(deltaV[:, 0] - deltaV[:, 3]), np.array([0, 1/8, 0, 3/8, 0, 3/8, 0, 1/8]))
            Dv_C0 = np.dot(np.abs(deltaV[:, 0] - deltaV[:, 3]), np.array([1/8, 0, 3/8, 0, 3/8, 0, 1/8, 0]))
            
        s = Dh_C1 + Dv_C1 + Dh_C0 + Dv_C0
        edge_factor = 0.5 if s == 0 else (Dh_C0 + Dh_C1)/s
        
        #Delta = deltaH[2:6, :] * (1-edge_factor) + deltaV[2:6, :].T * edge_factor

        weight_h = self.weight_cal(dx)
        weight_v = self.weight_cal(dy/2, gamma=1)

        deltaH_first = np.dot(deltaH, weight_h)
        deltaH_C0 = np.dot(deltaH_first, self.color_weight(weight_v, False))
        deltaH_C1 = np.dot(deltaH_first, self.color_weight(weight_v, True))
        
        weight_h = self.weight_cal(dx/2, gamma=1)
        weight_v = self.weight_cal(dy)
        
        deltaV_first = np.dot(deltaV, weight_v)
        deltaV_C0 = np.dot(deltaV_first, self.color_weight(weight_h, self.is_green(i, j)))
        deltaV_C1 = np.dot(deltaV_first, self.color_weight(weight_h, not self.is_green(i, j)))
        
        #Delta_C0 = deltaH_C0
        #Delta_C1 = deltaH_C1
        
        Delta_C0 = self.getDelta(deltaH_C0, deltaV_C0, Dh_C0, Dv_C0)
        Delta_C1 = self.getDelta(deltaH_C1, deltaV_C1, Dh_C1, Dv_C1)
        
        for m in range(-1, 3):
            for n in range(-1, 3):
                if not self.is_green(i+m, j+n):
                    Delta[m+1, n+1] = self._DeltaV(i+m, j+n) * edge_factor + self._DeltaH(i+m, j+n) * (1-edge_factor)
        
        i_near = round(i + dy)
        j_near = round(j + dx)
        
        if not self.is_green(i_near, j_near):
            Delta[int((dy-0.25)*2)+1, int((dx-0.25)*2)+1] = Delta_C0 if dy < 0.5 else Delta_C1
        
        return Delta, Delta_C0, Delta_C1, edge_factor
    
    def set_mode(self, i ,j):
        '''
        * mode 0: |B| G   * mode 1: |G| B
                   G  R              R  G
        
        * mode 2: |G| R   * mode 3: |R| G
                   B  G              G  B
        '''
        self.mode = (i % 2) << 1 | (j % 2)
    
    def get_position(self, oy, ox):
        scale_factor = 1/self.scale
        x = (ox + 0.5) * (scale_factor) - 0.5
        y = (oy + 0.5) * (scale_factor) - 0.5
        
        j = int(x)	#i = floor(x) 
        i = int(y)  #j = floor(y)
        dx = x - float(j)
        dy = y - float(i)
        
        return i, j, dy, dx

    def Delta(self):
        output_delta = np.zeros((self.outputHeight, self.outputWidth, 2), dtype=np.int16)
        
        for oy in range(10, self.outputHeight - 10):
            for ox in range(10, self.outputWidth - 10):
                
                i, j, dy, dx = self.get_position(oy, ox)
                self.set_mode(i, j)
                mode = self.mode
                det, Delta_C0, Delta_C1, edge = self._Delta(i, j, dy, dx)
                
                if mode == 0 or mode == 1:
                    output_delta[oy, ox, 1] = Delta_C0
                    output_delta[oy, ox, 0] = Delta_C1
                else:
                    output_delta[oy, ox, 0] = Delta_C0
                    output_delta[oy, ox, 1] = Delta_C1                             

        return output_delta
    
    def Algorithm(self):
        outputImage = np.zeros((self.outputHeight, self.outputWidth, 3), dtype=np.uint8)
        
        for oy in range(10, self.outputHeight - 10):
            for ox in range(10, self.outputWidth - 10):
                
                i, j, dy, dx = self.get_position(oy, ox)
                self.set_mode(i, j)
                mode = self.mode
                det, Delta_C0, Delta_C1, edge = self._Delta(i, j, dy, dx)

                a = self.weight_cal(dy, gamma=2.5-edge)
                b = self.weight_cal(dx, gamma=edge+1.5)
                
                arr = self.inputImage[i-1:i+3, j-1:j+3]
                float_arr = arr.astype(nb.float64)
                
                '''
                if mode == 0 or mode == 3:
                    det = np.array([[Delta_C1, 0 ,Delta_C1, 0],
                                    [0, Delta_C0, 0, Delta_C0],
                                    [Delta_C1, 0 ,Delta_C1, 0],
                                    [0, Delta_C0, 0, Delta_C0]], dtype=nb.float64)  
                else:
                    det = np.array([[0 ,Delta_C1, 0, Delta_C1],
                                    [Delta_C0, 0, Delta_C0, 0],
                                    [0 ,Delta_C1, 0, Delta_C1],
                                    [Delta_C0, 0, Delta_C0, 0]], dtype=nb.float64)
                '''
                '''
                if mode == 0 or mode == 3:
                    det[1, 1] = Delta_C0
                    det[2, 2] = Delta_C1
                else:
                    det[1, 2] = Delta_C0
                    det[2, 1] = Delta_C1
                '''
                float_arr = float_arr + det
                temp = np.dot(a.reshape(1,4), float_arr)
                epi  = np.dot(temp, b)
                
                tempG = np.sum(epi)
                
                if mode == 0:
                    tempB = tempG - Delta_C0
                    tempR = tempG - Delta_C1
                elif mode == 1:
                    tempB = tempG - Delta_C0
                    tempR = tempG - Delta_C1
                elif mode == 2:
                    tempR = tempG - Delta_C0
                    tempB = tempG - Delta_C1
                elif mode == 3:
                    tempR = tempG - Delta_C0
                    tempB = tempG - Delta_C1
                else:
                    tempR = 0
                    tempG = 0
                    tempB = 0

                outputImage[oy, ox, 1] = 255 if tempG > 255 else 0 if tempG < 0 else int(tempG)
                outputImage[oy, ox, 0] = 255 if tempR > 255 else 0 if tempR < 0 else int(tempR)
                outputImage[oy, ox, 2] = 255 if tempB > 255 else 0 if tempB < 0 else int(tempB)

                if False:#oy == 341 and ox == 272:
                    print(Delta_C0)
                    print(arr)
                    print(edge)
                    print(outputImage[oy, ox, 1])
                    print(float_arr)
                    print(dy, dx)
                    print(mode)

        return outputImage
        