import numpy as np
import numba as nb
from numba.experimental import jitclass
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
    
    def __weight_cal(self, vs, gamma=0):

        if vs == 1:
            return np.array([0.10151382, -0.30232316, 1.0731945, 0.15881203, -0.031327657])
        else:
            return np.array([-0.031327657, 0.15881203, 1.0731945, -0.30232316, 0.10151382])

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
        
    def getDelta(self, dh, dv, gh, gv, eh=0, ev=0):
        '''
        if np.abs(dh - dv) > 20:
            return -300#dh if eh/(gh*gh) > ev/(gv*gv) else dv
        '''
        s = gh + gv
        if 4*gh < gv:
            return dh
        elif 4*gv < gh:
            return dv
        #elif eh > 4*ev:
        #    return -300
        #elif ev > 4*eh:
        #    return 300
        #elif np.abs(dh - dv) > 20:
        #    return dh# if eh/(gh*gh) > ev/(gv*gv) else dv
        elif s == 0:
            return (dh + dv) / 2
        else:
            return (gh * dv + gv * dh) / s
        
    def __getDelta(self, dh, dv, gh, gv):
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
        Delta_green = np.zeros((4, 4))
        Delta_color = np.zeros((4, 4, 2))

        for m in range(-3, 5, 1):
            for n in range(-1, 3, 1):
                deltaH[m+3, n+1] = self._DeltaH(i+m, j+n)
                deltaV[m+3, n+1] = self._DeltaV(i+n, j+m)

        Dh = np.dot((np.abs(deltaH[2:6, 0] - deltaH[2:6, 2]) + np.abs(deltaH[2:6, 1] - deltaH[2:6, 3])), np.array([1/8, 3/8, 3/8, 1/8]))        
        Dhv = np.dot((np.abs(deltaH[2, :] - deltaH[4, :]) + np.abs(deltaH[3, :] - deltaH[5, :])), np.array([1/8, 3/8, 3/8, 1/8]))        

        Dv = np.dot((np.abs(deltaV[2:6, 0] - deltaV[2:6, 2]) + np.abs(deltaV[2:6, 1] - deltaV[2:6, 3])), np.array([1/8, 3/8, 3/8, 1/8]))
        Dvh = np.dot((np.abs(deltaH[2, :] - deltaH[4, :]) + np.abs(deltaH[3, :] - deltaH[5, :])), np.array([1/8, 3/8, 3/8, 1/8]))        
        
        Eh = Dhv if Dh == 0 else Dhv / Dh 
        Ev = Dvh if Dv == 0 else Dvh / Dv
        
        s = Dh + Dv
        edge_factor = 0.5 if s == 0 else Dh/s
        
        #Delta = deltaH[2:6, :] * (1-edge_factor) + deltaV[2:6, :].T * edge_factor

        ga = 2

        weight_h = self.weight_cal(dx ,gamma=ga*edge_factor)
        weight_v = self.weight_cal(dy/2, gamma=(ga*(1-edge_factor)))

        deltaH_first = np.dot(deltaH, weight_h)
        deltaH_C0 = np.dot(deltaH_first, self.color_weight(weight_v, False))
        deltaH_C1 = np.dot(deltaH_first, self.color_weight(weight_v, True))
        
        weight_h = self.weight_cal(dx/2, gamma=ga*edge_factor)
        weight_v = self.weight_cal(dy, gamma=ga*(1-edge_factor))
        
        deltaV_first = np.dot(deltaV, weight_v)
        deltaV_C0 = np.dot(deltaV_first, self.color_weight(weight_h, self.is_green(i, j)))
        deltaV_C1 = np.dot(deltaV_first, self.color_weight(weight_h, not self.is_green(i, j)))
        
        #Delta_C0 = deltaH_C0
        #Delta_C1 = deltaH_C1
        
        Delta_C0 = self.getDelta(deltaH_C0, deltaV_C0, Dh, Dv, Eh, Ev)
        Delta_C1 = self.getDelta(deltaH_C1, deltaV_C1, Dh, Dv, Eh, Ev)
        
        for m in range(-1, 3):
            for n in range(-1, 3):
                if not self.is_green(i+m, j+n):
                    Delta_green[m+1, n+1] = self.getDelta(self._DeltaH(i+m, j+n), self._DeltaV(i+m, j+n), Dh, Dv, Eh, Ev)
                    
        for m in range(-1, 3):
            for n in range(-1, 3):
                if self.is_green(i+m, j+n):
                    Delta_color[m+1, n+1, 1] = self._DeltaH(i+m, j+n)
                    Delta_color[m+1, n+1, 2] = self._DeltaV(i+m, j+n)
        
        '''
        i_near = round(i + dy)
        j_near = round(j + dx)
        
        if not self.is_green(i_near, j_near):
            Delta[int((dy-0.25)*2)+1, int((dx-0.25)*2)+1] = Delta_C0 if dy < 0.5 else Delta_C1
        '''
        return Delta_green, Delta_color, Delta_C0, Delta_C1, edge_factor
    
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
                det, dc, Delta_C0, Delta_C1, edge = self._Delta(i, j, dy, dx)
                
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
                det, det_c, Delta_C0, Delta_C1, edge = self._Delta(i, j, dy, dx)

                a = self.weight_cal(dy, gamma=5*(1-edge))#2.5-edge)
                b = self.weight_cal(dx, gamma=5*edge)#edge+1.5)
                
                orig = self.inputImage[i-1:i+3, j-1:j+3]
                mosaic = orig.astype(nb.float64)
                
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
                green = mosaic + det
                '''
                color_H = np.zeros((4, 4))
                color_V = np.zeros((4, 4))
                
                for i in range(4):
                    color_0[i ,:] = mosaic[i, :] + det_c[i, :, (i+1)%2]
                    color_1[i ,:] = mosaic[i, :] + det_c[i, :, (i+1)%2]
                epi_color = np.dot(color, b)
                '''
                epi_green = np.dot(np.dot(a.reshape(1,4), green), b)                
                
                tempG = np.sum(epi_green)
                
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
                '''
                if oy == 341 and ox == 309:
                    tempR = 0
                    tempG = 0
                    tempB = 0
                '''
                outputImage[oy, ox, 1] = 255 if tempG > 255 else 0 if tempG < 0 else int(tempG)
                outputImage[oy, ox, 0] = 255 if tempR > 255 else 0 if tempR < 0 else int(tempR)
                outputImage[oy, ox, 2] = 255 if tempB > 255 else 0 if tempB < 0 else int(tempB)

                if oy == 293 and ox == 310:
                    print("293, 310")
                    print(Delta_C0)
                    print(Delta_C1)
                    print(edge)
                    print(outputImage[oy, ox])
                    print(green)
                    print(dy, dx)
                    print(mode)
                    
                if oy == 294 and ox == 310:
                    print("294, 310")
                    print(Delta_C0)
                    print(Delta_C1)
                    print(edge)
                    print(outputImage[oy, ox])
                    print(green)
                    print(dy, dx)
                    print(mode)

        return outputImage
        