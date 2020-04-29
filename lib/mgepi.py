import numpy as np
import numba as nb
from numba import jitclass

spec = [
    ('width', nb.int64),
    ('height', nb.int64),
    ('outputWidth', nb.int64),
    ('outputHeight', nb.int64),
    ('scale', nb.float32),
    ('inputImage', nb.uint8[:, :]),
    ('outputImage', nb.uint8[:, :, :]),
    ('mode', nb.uint8),
    ('Dh_G', nb.float64),
    ('Dv_G', nb.float64)
]

@jitclass(spec)
class MGEPI:
    def __init__(self, oriImage, scale):
        self.width = oriImage.shape[1]
        self.height = oriImage.shape[0]
        self.scale = scale

        self.outputWidth = int(self.width * scale)
        self.outputHeight = int(self.height * scale)
        self.inputImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth, 3), 
                                    dtype=np.uint8)
        self.mode = 0
        self.Dh_G = 0
        self.Dv_G = 0
        
    def is_green(self, i, j):
        return (i%2) != (j%2)

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
                
    def getDelta(self, dh, dv, gh, gv):
        s = gh + gv
        if s == 0:
            return (dh + dv) / 2
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

    def DeltaH(self, i, j):
        Delta = np.zeros((3, 3))
        for m in range(0, 3):
            for n in range(0, 3):
                Delta[m, n] = self._DeltaH(i+2*(m-1), j+(n-1))
        
        tmp = np.dot(Delta, np.array([0.05, 0.9, 0.05]))
        Delta_H_total = np.dot(tmp, np.array([0.05, 0.9, 0.05]))
        Dh_G = np.dot(np.abs(Delta[:, 0] - Delta[:, 1])+np.abs(Delta[:, 1] - Delta[:, 2]), np.array([0.1, 0.8, 0.1]))
        
        return Delta_H_total, Dh_G
    
    def DeltaV(self, i, j):
        Delta = np.zeros((3, 3))
        for m in range(0, 3):
            for n in range(0, 3):
                Delta[m, n] = self._DeltaV(i+(m-1), j+2*(n-1))
        
        tmp = np.dot(np.array([0.05, 0.9, 0.05]), Delta)
        Delta_V_total = np.dot(tmp, np.array([0.05, 0.9, 0.05]))
        Dv_G = np.dot(np.abs(Delta[0, :] - Delta[1, :])+np.abs(Delta[1, :] - Delta[2, :]), np.array([0.1, 0.8, 0.1]))
        
        return Delta_V_total, Dv_G
    
    def mgbi(self, i, j):
        inputImage = self.inputImage
        outImage = np.zeros(3, dtype=np.uint8)
        
        '''
        * mode 0: |B| G   * mode 1: |G| B
                   G  R              R  G
        
        * mode 2: |G| R   * mode 3: |R| G
                   B  G              G  B
        '''
        self.mode = int(i % 2) * 2 + int(j % 2)

        if self.mode == 0 or self.mode == 3:
            # 補G #
                # 色差總和, 方向梯度 #
            Delta_H_total, self.Dh_G = self.DeltaH(i, j)
            Delta_V_total, self.Dv_G = self.DeltaV(i, j)
            
                # 插補出G #
            det = self.getDelta(Delta_H_total, Delta_V_total, self.Dh_G, self.Dv_G)
            temp = inputImage[i, j] + det
            
            outImage[1] = 255 if temp > 255 else 0 if temp < 0 else temp

            # 補G #
            # 補RB #
                # 估計尚未插補的G值 #
                    # 左上 #
            Delta_H0 = self._DeltaH(i-1, j-1)
            Delta_V0 = self._DeltaV(i-1, j-1)
                    # 右上 #
            Delta_H1 = self._DeltaH(i-1, j+1)
            Delta_V1 = self._DeltaV(i-1, j+1)
                    # 左下 #
            Delta_H2 = self._DeltaH(i+1, j-1)
            Delta_V2 = self._DeltaV(i+1, j-1)
                    # 右下 #
            Delta_H3 = self._DeltaH(i+1, j+1)
            Delta_V3 = self._DeltaV(i+1, j+1)

            Delta_H_total = (Delta_H0 + Delta_H1 + Delta_H2 + Delta_H3) / 4
            Delta_V_total = (Delta_V0 + Delta_V1 + Delta_V2 + Delta_V3) / 4
            Dh_O = (abs(Delta_H0 - Delta_H1) + abs(Delta_H2 - Delta_H3)) / 2
            Dv_O = (abs(Delta_V0 - Delta_V2) + abs(Delta_V1 - Delta_V3)) / 2

            det = self.getDelta(Delta_H_total, Delta_V_total, Dh_O, Dv_O)
            temp = outImage[1] - det

            outImage[(0 if self.mode == 0 else 2)] = 255 if temp > 255 else 0 if temp < 0 else temp
            outImage[(2 if self.mode == 0 else 0)] = inputImage[i, j]
            # 補RB #
            
        else:           #  odd pixel G
            # 補G 上 RB #
                # 估計B的G值 #
                    # 左 #
            Delta_H1 = self._DeltaH(i, j-1)
            Delta_V1 = self._DeltaV(i, j-1)
                    # 右 #
            Delta_H2 = self._DeltaH(i, j+1)
            Delta_V2 = self._DeltaV(i, j+1)

            Delta_H1_total = (Delta_H1 + Delta_H2) / 2
            Delta_V1_total = (Delta_V1 + Delta_V2) / 2

            det = self.getDelta(Delta_H1_total, Delta_V1_total, self.Dh_G, self.Dv_G)
            temp = inputImage[i, j] - det
            
            outImage[(2 if self.mode == 1 else 0)] = 255 if temp > 255 else 0 if temp < 0 else temp
            
                # 估計R的G值 #
                    # 上 #
            Delta_H1 = self._DeltaH(i-1, j)
            Delta_V1 = self._DeltaV(i-1, j)
                    # 下 #
            Delta_H2 = self._DeltaH(i+1, j)
            Delta_V2 = self._DeltaV(i+1, j)

            Delta_H1_total = (Delta_H1 + Delta_H2) / 2
            Delta_V1_total = (Delta_V1 + Delta_V2) / 2

            det = self.getDelta(Delta_H1_total, Delta_V1_total, self.Dh_G, self.Dv_G)
            temp = inputImage[i, j] - det
            
            outImage[(0 if self.mode == 1 else 2)] = 255 if temp > 255 else 0 if temp < 0 else temp
            outImage[1] = inputImage[i, j]
            # 補G 上 RB #
        
        s = self.Dh_G + self.Dv_G
        edge_factor = 0.5 if s == 0 else self.Dh_G / s

        return outImage, edge_factor
    
    def weight_cal(self, vs, gamma=2):
        x_gamma = gamma/2
        
        tmp = (1-vs)
        vc3 = (tmp*vs*vs)*x_gamma
        vc0 = (tmp*tmp*vs)*x_gamma
        vc1 = tmp + 2*vc0 - 1*vc3
        vc2 = vs + 2*vc3 - 1*vc0
        vc0 = (-1)*vc0
        vc3 = (-1)*vc3

        return [vc0, vc1, vc2, vc3]

    def Algorithm(self):
        tempWindow = np.zeros((5, 5, 3), dtype=np.uint8)
        edge = np.zeros((5, 5))
        for oy in range(self.outputHeight - 7):
            for ox in range(5, self.outputWidth - 5):

                j = int((ox)/2)
                i = int((oy)/2)
                
                for m in range(5):
                    for n in range(5):
                        tempWindow[m, n], edge[m, n] = self.mgbi(i-2+m, j-2+n)
                        
                #e = np.sum(edge[1:3, 1:3])/4
                
                odd = np.array([0.10151382, -0.30232316, 1.0731945, 0.15881203, -0.031327657])#self.weight_cal(dy, 2.5-e))
                eve = np.array([-0.031327657, 0.15881203, 1.0731945, -0.30232316, 0.10151382])#self.weight_cal(dx, e+1.5))
                
                b = odd if (ox % 2) == 1 else eve
                a = odd if (oy % 2) == 1 else eve
                
                for color in range(3):
                    ori = tempWindow[:, :, color].astype(nb.float64)
                    temp = round(np.dot(np.dot(a, ori), b))
                    
                    temp *= (temp>0)
                    if temp > 255 :
                        temp = 255

                    self.outputImage[oy, ox, color] = temp

        return self.outputImage
