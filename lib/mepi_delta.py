import numpy as np
import numba as nb

spec = [
    ('width', nb.int64),
    ('height', nb.int64),
    ('outputWidth', nb.int64),
    ('outputHeight', nb.int64),
    ('inputImage', nb.uint8[:, :]),
    ('outputImage', nb.int16[:, :, :]),
    ('scale', nb.float64),
    ('mode', nb.uint8),
]

@nb.jitclass(spec)    
class MEPR:
    def __init__(self, oriImage, scale):
        self.width = oriImage.shape[1]
        self.height = oriImage.shape[0]

        self.scale = scale
        self.outputWidth = int(self.width * scale)
        self.outputHeight = int(self.height * scale)
        self.inputImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth, 2), dtype=np.int16)
        
    def weight_cal(self, vs):
        gamma = 2
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

    def _DeltaH(self, i, j, isGreen=False):
        if isGreen == True:
            return (self.inputImage[i, j] / 2) + (self.inputImage[i, j-2] / 4) +   \
                (self.inputImage[i, j+2] / 4) - (self.inputImage[i, j-1] / 2) - \
                (self.inputImage[i, j+1] / 2)
        else:
            return (self.inputImage[i, j-1] / 2) + (self.inputImage[i, j+1] / 2) - \
                (self.inputImage[i, j-2] / 4) - (self.inputImage[i, j+2] / 4) - \
                (self.inputImage[i, j] / 2)

    def _DeltaV(self, i, j, isGreen=False):
        if isGreen == True:
            return (self.inputImage[i, j] / 2) + (self.inputImage[i-2, j] / 4) +   \
                (self.inputImage[i+2, j] / 4) - (self.inputImage[i-1, j] / 2) - \
                (self.inputImage[i+1, j] / 2)
        else:
            return (self.inputImage[i-1, j] / 2) + (self.inputImage[i+1, j] / 2) - \
                (self.inputImage[i-2, j] / 4) - (self.inputImage[i+2, j] / 4) - \
                (self.inputImage[i, j] / 2)

    def DeltaH(self, i, j, dx, dy, isGreen, ioffset=0, joffset=0):
        delta = np.empty((4, 4))

        for m in range(4):
            for n in range(4):
                delta[m, n] = self._DeltaH(i+(m-ioffset-1)*2, j+(n-joffset-1), isGreen==bool((n-joffset)%2))

        a = self.weight_cal(dx)
        b = self.weight_cal(dy/2)

        delta_total = np.dot(delta, a)
        Delta_H_total = np.dot(delta_total, b)
        Dh_G = np.dot(np.abs(delta[:, 0] - delta[:, 3]), np.array([1/8,3/8,3/8,1/8]))

        return Delta_H_total, Dh_G

    def DeltaV(self, i, j, dy, dx, isGreen, ioffset=0, joffset=0):
        delta = np.empty((4, 4))

        for m in range(4):
            for n in range(4):
                delta[m,n] = self._DeltaV(i+(n-ioffset-1), j+(m-joffset-1)*2, isGreen==bool((n-ioffset)%2))
        
        a = self.weight_cal(dy)
        b = self.weight_cal(dx/2)
        
        delta_total = np.dot(delta, a)
        Delta_V_total = np.dot(delta_total, b)
        Dv_G = np.dot(np.abs(delta[:, 0] - delta[:, 3]), np.array([1/8,3/8,3/8,1/8]))

        return Delta_V_total, Dv_G
    
    def getDelta(self, dh, dv, gh, gv):
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

    def Algorithm(self):
        scale_factor_h = 0.5 #scale_factor_h的倒數 
        scale_factor_v = 0.5#scale_factor_v的倒數 
        
        for oy in range(10, self.outputHeight - 10):
            for ox in range(10, self.outputWidth - 10):
                x = (ox + 0.5) * (scale_factor_h) - 0.5
                y = (oy + 0.5) * (scale_factor_v) - 0.5
                
                j = int(x)	#i = floor(x) 
                i = int(y)  #j = floor(y)
                dx = x - float(j)
                dy = y - float(i)

                '''
                * mode 0: |B| G   * mode 1: |G| B
                           G  R              R  G
                
                * mode 2: |G| R   * mode 3: |R| G
                           B  G              G  B
                '''
                self.mode = (i % 2) << 1 | (j % 2)

                if self.mode == 0 or self.mode == 3:
                    Delta_H_C0, Dh_C0 = self.DeltaH(i, j, dx, dy, False)
                    Delta_V_C0, Dv_C0 = self.DeltaV(i, j, dy, dx, False)                

                    Delta_H_C1, Dh_C1 = self.DeltaH(i+1, j, dx, 1-dy, True)
                    Delta_V_C1, Dv_C1 = self.DeltaV(i, j+1, dy, 1-dx, True)                
                
                else:
                    Delta_H_C0, Dh_C0 = self.DeltaH(i, j, dx, dy, True)
                    Delta_V_C1, Dv_C1 = self.DeltaV(i, j, dy, dx, True)                

                    Delta_H_C1, Dh_C1 = self.DeltaH(i+1, j, dx, 1-dy, False)
                    Delta_V_C0, Dv_C0 = self.DeltaV(i, j+1, dy, 1-dx, False)
                    
                Delta_C0 = self.getDelta(Delta_H_C0, Delta_V_C0, Dh_C0, Dv_C0)
                Delta_C1 = self.getDelta(Delta_H_C1, Delta_V_C1, Dh_C1, Dv_C1)
                
                if self.mode == 0 or self.mode == 1:
                    self.outputImage[oy, ox, 1] = Delta_C0
                    self.outputImage[oy, ox, 0] = Delta_C1
                else:
                    self.outputImage[oy, ox, 0] = Delta_C0
                    self.outputImage[oy, ox, 1] = Delta_C1                             

        return self.outputImage
        