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

    def DeltaH(self, i, j):
        c0 = self.mode == 1 or self.mode == 2
        c1 = not c0

        # 求水平色差 #                
                # 中 #
        Delta_H0 = self._DeltaH(i, j-1, c1)
        Delta_H1 = self._DeltaH(i, j, c0)
        Delta_H2 = self._DeltaH(i, j+1, c1)
        Delta_H20 = self._DeltaH(i, j+2, c0)
                # 上 #
        Delta_H3 = self._DeltaH(i-2, j-1, c1)
        Delta_H4 = self._DeltaH(i-2, j, c0)
        Delta_H5 = self._DeltaH(i-2, j+1, c1)
        Delta_H50 = self._DeltaH(i-2, j+2, c0)
                # 下 #
        Delta_H6 = self._DeltaH(i+2, j-1, c1)
        Delta_H7 = self._DeltaH(i+2, j, c0)
        Delta_H8 = self._DeltaH(i+2, j+1, c1)
        Delta_H80 = self._DeltaH(i+2, j+2, c0)

        Delta_H1_total = (-0.125 * Delta_H0 + 0.625 * Delta_H1 + 0.625 * Delta_H2 -0.125 * Delta_H20)
        Delta_H2_total = (-0.125 * Delta_H3 + 0.625 * Delta_H4 + 0.625 * Delta_H5 -0.125 * Delta_H50)
        Delta_H3_total = (-0.125 * Delta_H6 + 0.625 * Delta_H7 + 0.625 * Delta_H8 -0.125 * Delta_H80)

        Delta_H_total = (Delta_H2_total + 2 * Delta_H1_total + Delta_H3_total) / 4
        Dh_G = (abs(Delta_H3 - Delta_H50) + 2 * abs(Delta_H0 - Delta_H20) + abs(Delta_H6 - Delta_H80)) / 4

        return Delta_H_total, Dh_G

    def DeltaV(self, i, j):
        c0 = self.mode == 1 or self.mode == 2
        c1 = not c0
        
        # 求垂直色差
                # 中 #
        Delta_V0 = self._DeltaV(i-1, j, c1)
        Delta_V1 = self._DeltaV(i, j, c0)
        Delta_V2 = self._DeltaV(i+1, j, c1)
        Delta_V20 = self._DeltaV(i+2, j, c0)
                # 左 #
        Delta_V3 = self._DeltaV(i-1, j-2, c1)
        Delta_V4 = self._DeltaV(i, j-2, c0)
        Delta_V5 = self._DeltaV(i+1, j-2, c1)
        Delta_V50 = self._DeltaV(i+2, j-2, c0)
                #右 #
        Delta_V6 = self._DeltaV(i-1, j+2, c1)
        Delta_V7 = self._DeltaV(i, j+2, c0)
        Delta_V8 = self._DeltaV(i+1, j+2, c1)
        Delta_V80 = self._DeltaV(i+2, j+2, c0)

        Delta_V1_total = (-0.125 * Delta_V0 + 0.625 * Delta_V1 + 0.625 * Delta_V2 -0.125 * Delta_V20)
        Delta_V2_total = (-0.125 * Delta_V3 + 0.625 * Delta_V4 + 0.625 * Delta_V5 -0.125 * Delta_V50)
        Delta_V3_total = (-0.125 * Delta_V6 + 0.625 * Delta_V7 + 0.625 * Delta_V8 -0.125 * Delta_V80)

        Delta_V_total = (Delta_V2_total + 2 * Delta_V1_total + Delta_V3_total) / 4
        Dv_G = (abs(Delta_V3 - Delta_V50) + 2 * abs(Delta_V0 - Delta_V20) + abs(Delta_V6 - Delta_V80)) / 4

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
            Delta_H_total, Dh_G = self.DeltaH(i, j)
            Delta_V_total, Dv_G = self.DeltaV(i, j)
            
                # 插補出G #
            if Dv_G <= Dh_G:
                if Dv_G * 4 <= Dh_G:
                    temp = inputImage[i, j] + Delta_V_total
                elif Dv_G * 2 <= Dh_G:
                    temp = inputImage[i, j] + (3 * Delta_V_total + Delta_H_total) / 4
                else:
                    temp = inputImage[i, j] + (Delta_V_total + Delta_H_total) / 2
            else:
                if Dh_G * 4 <= Dv_G:
                    temp = inputImage[i, j] + Delta_H_total
                elif Dh_G * 2 <= Dv_G:
                    temp = inputImage[i, j] + (3 * Delta_H_total + Delta_V_total) / 4
                else:
                    temp = inputImage[i, j] + (Delta_H_total + Delta_V_total) / 2

            outImage[1] = 255 if temp > 255 else 0 if temp < 0 else temp

            # 補G #
            # 補RB #
                # 估計尚未插補的G值 #
                    # 左上 #
            Delta_H0 = self._DeltaH(i-1, j-1, False)
            Delta_V0 = self._DeltaV(i-1, j-1, False)
                    # 右上 #
            Delta_H1 = self._DeltaH(i-1, j+1, False)
            Delta_V1 = self._DeltaV(i-1, j+1, False)
                    # 左下 #
            Delta_H2 = self._DeltaH(i+1, j-1, False)
            Delta_V2 = self._DeltaV(i+1, j-1, False)
                    # 右下 #
            Delta_H3 = self._DeltaH(i+1, j+1, False)
            Delta_V3 = self._DeltaV(i+1, j+1, False)

            Delta_H_total = (Delta_H0 + Delta_H1 + Delta_H2 + Delta_H3) / 4
            Delta_V_total = (Delta_V0 + Delta_V1 + Delta_V2 + Delta_V3) / 4
            Dh_O = (abs(Delta_H0 - Delta_H1) + abs(Delta_H2 - Delta_H3)) / 2
            Dv_O = (abs(Delta_V0 - Delta_V2) + abs(Delta_V1 - Delta_V3)) / 2

            if Dv_O <= Dh_O:
                if Dv_O * 4 <= Dh_O:
                    temp = outImage[1] - Delta_V_total
                elif Dv_O * 2 <= Dh_O:
                    temp = outImage[1] - (3 * Delta_V_total + Delta_H_total) / 4
                else:
                    temp = outImage[1] - (Delta_V_total + Delta_H_total) / 2
            
            else:
                if Dh_O * 4 <= Dv_O:
                    temp = outImage[1] - Delta_H_total
                elif Dh_O * 2 <= Dv_O:
                    temp = outImage[1] - (3 * Delta_H_total + Delta_V_total) / 4
                else:
                    temp = outImage[1] - (Delta_H_total + Delta_V_total) / 2                    

            outImage[(0 if self.mode == 0 else 2)] = 255 if temp > 255 else 0 if temp < 0 else temp
            outImage[(2 if self.mode == 0 else 0)] = inputImage[i, j]
            # 補RB #
            
        else:           #  odd pixel G
            # 補G 上 RB #
                # 估計B的G值 #
                    # 左 #
            Delta_H1 = self._DeltaH(i, j-1, False)
            Delta_V1 = self._DeltaV(i, j-1, False)
                    # 右 #
            Delta_H2 = self._DeltaH(i, j+1, False)
            Delta_V2 = self._DeltaV(i, j+1, False)

            Delta_H1_total = (Delta_H1 + Delta_H2) / 2
            Delta_V1_total = (Delta_V1 + Delta_V2) / 2

            if Dv_G <= Dh_G:                    
                if Dv_G * 4 <= Dh_G:
                    temp = inputImage[i, j] - Delta_V1_total
                elif Dv_G * 2 <= Dh_G:
                    temp = inputImage[i, j] - (3 * Delta_V1_total + Delta_H1_total) / 4
                else:
                    temp = inputImage[i, j] - (Delta_V1_total + Delta_H1_total) / 2
            
            else:                    
                if Dh_G * 4 <= Dv_G:
                    temp = inputImage[i, j] - Delta_H1_total
                elif Dh_G * 2 <= Dv_G:
                    temp = inputImage[i, j] - (3 * Delta_H1_total + Delta_V1_total) / 4
                else:
                    temp = inputImage[i, j] - (Delta_H1_total + Delta_V1_total) / 2
            
            outImage[(2 if self.mode == 1 else 0)] = 255 if temp > 255 else 0 if temp < 0 else temp
            
                # 估計R的G值 #
                    # 上 #
            Delta_H1 = self._DeltaH(i-1, j, False)
            Delta_V1 = self._DeltaV(i-1, j, False)
                    # 下 #
            Delta_H2 = self._DeltaH(i+1, j, False)
            Delta_V2 = self._DeltaV(i+1, j, False)

            Delta_H1_total = (Delta_H1 + Delta_H2) / 2
            Delta_V1_total = (Delta_V1 + Delta_V2) / 2

            if Dv_G <= Dh_G:                    
                if Dv_G * 4 <= Dh_G:
                    temp = inputImage[i, j] - Delta_V1_total
                elif Dv_G * 2 <= Dh_G:
                    temp = inputImage[i, j] - (3 * Delta_V1_total + Delta_H1_total) / 4
                else:
                    temp = inputImage[i, j] - (Delta_V1_total + Delta_H1_total) / 2
            
            else:                    
                if Dh_G * 4 <= Dv_G:
                    temp = inputImage[i, j] - Delta_H1_total
                elif Dh_G * 2 <= Dv_G:
                    temp = inputImage[i, j] - (3 * Delta_H1_total + Delta_V1_total) / 4
                else:
                    temp = inputImage[i, j] - (Delta_H1_total + Delta_V1_total) / 2
            
            outImage[(0 if self.mode == 1 else 2)] = 255 if temp > 255 else 0 if temp < 0 else temp
            outImage[1] = inputImage[i, j]
            # 補G 上 RB #

        return outImage
    
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

    def Algorithm(self):
        tempWindow = np.zeros((4, 4, 3), dtype=np.uint8)
        for oy in range(self.outputHeight - 7):
            for ox in range(5, self.outputWidth - 5):
                x = (ox + 0.5) * (1/self.scale) - 0.5
                y = (oy + 0.5) * (1/self.scale) - 0.5
                
                j = int(x)	#i = floor(x) 
                i = int(y) #j = floor(y)
                dx = x - float(j)
                dy = y - float(i)
                
                for m in range(4):
                    for n in range(4):
                        tempWindow[m, n] = self.mgbi(i-1+m, j-1+n)
                        
                a1, a2, a3, a4 = self.weight_cal(dy)
                b1, b2, b3, b4 = self.weight_cal(dx)

                for color in range(3):
                    in1 = tempWindow[0, 0, color] * a1 + tempWindow[1, 0, color] * a2 + \
                        tempWindow[2, 0, color] * a3 + tempWindow[3, 0, color] * a4
                        
                    in2 = tempWindow[0, 1, color] * a1 + tempWindow[1, 1, color] * a2 + \
                        tempWindow[2, 1, color] * a3 + tempWindow[3, 1, color] * a4
                    
                    in3 = tempWindow[0, 2, color] * a1 + tempWindow[1, 2, color] * a2 + \
                        tempWindow[2, 2, color] * a3 + tempWindow[3, 2, color] * a4
                    
                    in4 = tempWindow[0, 3, color] * a1 + tempWindow[1, 3, color] * a2 + \
                        tempWindow[2, 3, color] * a3 + tempWindow[3, 3, color] * a4       
                    
                    temp = round(in1*b1 + in2*b2 + in3*b3 + in4*b4)
                
                    temp *= (temp>0)
                    if temp > 255 :
                        temp = 255

                    self.outputImage[oy, ox, color] = temp

        return self.outputImage
