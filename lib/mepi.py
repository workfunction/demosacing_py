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
    ('outputImage', uint8[:, :, :]),
    ('scale', float64),
    ('mode', uint8),
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
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth, 3), dtype=np.uint8)
        self.mode = 0
        
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

        return vc0, vc1, vc2, vc3
    
    def weight_cal3(self, d):
        r = 4 + d * 4 - d * d
        vc1 = ((2 + d) * (2 - d))/r
        vc0 = (d * (2 + d))/r
        vc2 = (d * (2 - d))/r

        return vc0, vc1, vc2

    def _DeltaH(self, i, j, isGreen):
        if isGreen == True:
            return (self.inputImage[i, j] / 2) + (self.inputImage[i, j-2] / 4) +   \
                (self.inputImage[i, j+2] / 4) - (self.inputImage[i, j-1] / 2) - \
                (self.inputImage[i, j+1] / 2)
        else:
            return (self.inputImage[i, j-1] / 2) + (self.inputImage[i, j+1] / 2) - \
                (self.inputImage[i, j-2] / 4) - (self.inputImage[i, j+2] / 4) - \
                (self.inputImage[i, j] / 2)

    def _DeltaV(self, i, j, isGreen):
        if isGreen == True:
            return (self.inputImage[i, j] / 2) + (self.inputImage[i-2, j] / 4) +   \
                (self.inputImage[i+2, j] / 4) - (self.inputImage[i-1, j] / 2) - \
                (self.inputImage[i+1, j] / 2)
        else:
            return (self.inputImage[i-1, j] / 2) + (self.inputImage[i+1, j] / 2) - \
                (self.inputImage[i-2, j] / 4) - (self.inputImage[i+2, j] / 4) - \
                (self.inputImage[i, j] / 2)

    def DeltaH(self, i, j, dx, dy, isGreen):
        c0 = isGreen
        c1 = not c0

        # 求水平色差 #                
                # 中 #
        Delta_H0 = self._DeltaH(i, j-1, c1)
        Delta_H1 = self._DeltaH(i, j, c0)
        Delta_H2 = self._DeltaH(i, j+1, c1)
        Delta_H3 = self._DeltaH(i, j+2, c0)
                # 上 #
        Delta_H4 = self._DeltaH(i-2, j-1, c1)
        Delta_H5 = self._DeltaH(i-2, j, c0)
        Delta_H6 = self._DeltaH(i-2, j+1, c1)
        Delta_H7 = self._DeltaH(i-2, j+2, c0)
                # 下 #
        Delta_H8 = self._DeltaH(i+2, j-1, c1)
        Delta_H9 = self._DeltaH(i+2, j, c0)
        Delta_H10 = self._DeltaH(i+2, j+1, c1)
        Delta_H11 = self._DeltaH(i+2, j+2, c0)

        a0, a1, a2, a3 = self.weight_cal(dx)

        Delta_H1_total = (a0 * Delta_H0 + a1 * Delta_H1 + a2 * Delta_H2 + a3 * Delta_H3)
        Delta_H2_total = (a0 * Delta_H4 + a1 * Delta_H5 + a2 * Delta_H6 + a3 * Delta_H7)
        Delta_H3_total = (a0 * Delta_H8 + a1 * Delta_H9 + a2 * Delta_H10 + a3 * Delta_H11)

        b0, b1, b2 = self.weight_cal3(dy)

        Delta_H_total = b0 * Delta_H2_total + b1 * Delta_H1_total + b2 * Delta_H3_total  
        Dh_G = (abs(Delta_H4 - Delta_H7) + 2 * abs(Delta_H0 - Delta_H3) + abs(Delta_H8 - Delta_H11)) / 4

        return Delta_H_total, Dh_G

    def DeltaV(self, i, j, dy, dx, isGreen):
        c0 = isGreen
        c1 = not c0

        # 求垂直色差
                # 中 #
        Delta_V0 = self._DeltaV(i-1, j, c1)
        Delta_V1 = self._DeltaV(i, j, c0)
        Delta_V2 = self._DeltaV(i+1, j, c1)
        Delta_V3 = self._DeltaV(i+2, j, c0)
                # 左 #
        Delta_V4 = self._DeltaV(i-1, j-2, c1)
        Delta_V5 = self._DeltaV(i, j-2, c0)
        Delta_V6 = self._DeltaV(i+1, j-2, c1)
        Delta_V7 = self._DeltaV(i+2, j-2, c0)
                #右 #
        Delta_V8 = self._DeltaV(i-1, j+2, c1)
        Delta_V9 = self._DeltaV(i, j+2, c0)
        Delta_V10 = self._DeltaV(i+1, j+2, c1)
        Delta_V11 = self._DeltaV(i+2, j+2, c0)

        a0, a1, a2, a3 = self.weight_cal(dy)

        Delta_V1_total = (a0 * Delta_V0 + a1 * Delta_V1 + a2 * Delta_V2 + a3 * Delta_V3)
        Delta_V2_total = (a0 * Delta_V4 + a1 * Delta_V5 + a2 * Delta_V6 + a3 * Delta_V7)
        Delta_V3_total = (a0 * Delta_V8 + a1 * Delta_V9 + a2 * Delta_V10 + a3 * Delta_V11)

        b0, b1, b2 = self.weight_cal3(dx)

        Delta_V_total = (b0 * Delta_V2_total + b1 * Delta_V1_total + b2 * Delta_V3_total)
        Dv_G = (abs(Delta_V4 - Delta_V7) + 2 * abs(Delta_V0 - Delta_V3) + abs(Delta_V8 - Delta_V11)) / 4

        return Delta_V_total, Dv_G

    def Algorithm(self):
        scale_factor_h = 0.5 #scale_factor_h的倒數 
        scale_factor_v = 0.5#scale_factor_v的倒數 
        
        for oy in range(1, self.outputHeight - 3):
            for ox in range(1, self.outputWidth - 3):
                x = (ox + 0.5) * (scale_factor_h) - 0.5
                y = (oy + 0.5) * (scale_factor_v) - 0.5
                
                j = int(x)	#i = floor(x) 
                i = int(y) #j = floor(y)
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
                    
                if Dv_C0 <= Dh_C0:
                    if Dv_C0 * 4 <= Dh_C0:
                        Delta_C0 = Delta_V_C0
                    elif Dv_C0 * 2 <= Dh_C0:
                        Delta_C0 = (3 * Delta_V_C0 + Delta_H_C0) / 4
                    else:
                        Delta_C0 = (Delta_V_C0 + Delta_H_C0) / 2
                else:
                    if Dh_C0 * 4 <= Dv_C0:
                        Delta_C0 = Delta_H_C0
                    elif Dh_C0 * 2 <= Dv_C0:
                        Delta_C0 = (3 * Delta_H_C0 + Delta_V_C0) / 4
                    else:
                        Delta_C0 = (Delta_H_C0 + Delta_V_C0) / 2
                
                if Dv_C1 <= Dh_C1:
                    if Dv_C1 * 4 <= Dh_C1:
                        Delta_C1 = Delta_V_C1
                    elif Dv_C1 * 2 <= Dh_C1:
                        Delta_C1 = (3 * Delta_V_C1 + Delta_H_C1) / 4
                    else:
                        Delta_C1 = (Delta_V_C1 + Delta_H_C1) / 2
                else:
                    if Dh_C1 * 4 <= Dv_C1:
                        Delta_C1 = Delta_H_C1
                    elif Dh_C1 * 2 <= Dv_C1:
                        Delta_C1 = (3 * Delta_H_C1 + Delta_V_C1) / 4
                    else:
                        Delta_C1 = (Delta_H_C1 + Delta_V_C1) / 2

                if self.mode == 0:
                    tempG = \
                        self.inputImage[i, j] / 2 + \
                        self.inputImage[i+1, j+1] / 2 + \
                        Delta_C0 / 2 + Delta_C1 / 2
                    tempG = 255 if tempG > 255 else 0 if tempG < 0 else tempG
                    tempB = tempG - Delta_C0
                    tempR = tempG - Delta_C1
                elif self.mode == 1:
                    tempG = \
                        self.inputImage[i, j+1] / 2 + \
                        self.inputImage[i+1, j] / 2 + \
                        Delta_C0 / 2 + Delta_C1 / 2
                    tempG = 255 if tempG > 255 else 0 if tempG < 0 else tempG
                    tempB = tempG - Delta_C0
                    tempR = tempG - Delta_C1
                elif self.mode == 2:
                    tempG = \
                        self.inputImage[i, j+1] / 2 + \
                        self.inputImage[i+1, j] / 2 + \
                        Delta_C0 / 2 + Delta_C1 / 2
                    tempG = 255 if tempG > 255 else 0 if tempG < 0 else tempG
                    tempR = tempG - Delta_C0
                    tempB = tempG - Delta_C1
                elif self.mode == 3:
                    tempG = \
                        self.inputImage[i, j] / 2 + \
                        self.inputImage[i+1, j+1] / 2 + \
                        Delta_C0 / 2 + Delta_C1 / 2
                    tempG = 255 if tempG > 255 else 0 if tempG < 0 else tempG
                    tempR = tempG - Delta_C0
                    tempB = tempG - Delta_C1
                else:
                    tempR = 0
                    tempG = 0
                    tempB = 0

                self.outputImage[oy, ox, 1] = tempG
                self.outputImage[oy, ox, 0] = 255 if tempR > 255 else 0 if tempR < 0 else tempR
                self.outputImage[oy, ox, 2] = 255 if tempB > 255 else 0 if tempB < 0 else tempB
                                    

        return self.outputImage
        