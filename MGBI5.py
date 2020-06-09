import numpy as np
import numba as nb
from numba import jitclass
from numba import int64
from numba import uint8

spec = [
    ('width', int64),               # a simple scalar field
    ('height', int64),          # an array field
    ('outputWidth', int64),
    ('outputHeight', int64),
    ('inputImage', uint8[:, :]),
    ('outputImage', uint8[:, :, :]),
    ('mode', uint8),
]

@jitclass(spec)
class MGBI_5:
    def DeltaHG(self, i, j):
        return (self.inputImage[i, j] / 2) + (self.inputImage[i, j-2] / 4) + (self.inputImage[i, j+2] / 4) - (self.inputImage[i, j-1] / 2) - (self.inputImage[i, j+1] / 2)

    def DeltaHRB(self, i, j):
        return (self.inputImage[i, j-1] / 2) + (self.inputImage[i, j+1] / 2) - (self.inputImage[i, j-2] / 4) - (self.inputImage[i, j+2] / 4) - (self.inputImage[i, j] / 2)

    def DeltaVG(self, i, j):
        return (self.inputImage[i, j] / 2) + (self.inputImage[i-2, j] / 4) + (self.inputImage[i+2, j] / 4) - (self.inputImage[i-1, j] / 2) - (self.inputImage[i+1, j] / 2)

    def DeltaVRB(self, i, j):
        return (self.inputImage[i-1, j] / 2) + (self.inputImage[i+1, j] / 2) - (self.inputImage[i-2, j] / 4) - (self.inputImage[i+2, j] / 4) - (self.inputImage[i, j] / 2)

    def DeltaH(self, i, j):
        # 求水平色差 #                
        if self.mode == 0:
                    # 中 #
            Delta_H0 = self.DeltaHG(i, j-1)
            Delta_H1 = self.DeltaHRB(i, j)
            Delta_H2 = self.DeltaHG(i, j+1)
                    # 上 #
            Delta_H3 = self.DeltaHG(i-2, j-1)
            Delta_H4 = self.DeltaHRB(i-2, j)
            Delta_H5 = self.DeltaHG(i-2, j+1)
                    # 下 #
            Delta_H6 = self.DeltaHG(i+2, j-1)
            Delta_H7 = self.DeltaHRB(i+2, j)
            Delta_H8 = self.DeltaHG(i+2, j+1)
        else:
                    # 中 #
            Delta_H0 = self.DeltaHRB(i, j-1)
            Delta_H1 = self.DeltaHG(i, j)
            Delta_H2 = self.DeltaHRB(i, j+1)
                    # 上 #
            Delta_H3 = self.DeltaHRB(i-2, j-1)
            Delta_H4 = self.DeltaHG(i-2, j)
            Delta_H5 = self.DeltaHRB(i-2, j+1)
                    # 下 #
            Delta_H6 = self.DeltaHRB(i+2, j-1)
            Delta_H7 = self.DeltaHG(i+2, j)
            Delta_H8 = self.DeltaHRB(i+2, j+1)

        Delta_H1_total = (Delta_H0 + 2 * Delta_H1 + Delta_H2) / 4
        Delta_H2_total = (Delta_H3 + 2 * Delta_H4 + Delta_H5) / 4
        Delta_H3_total = (Delta_H6 + 2 * Delta_H7 + Delta_H8) / 4

        Delta_H_total = (Delta_H2_total + 2 * Delta_H1_total + Delta_H3_total) / 4
        Dh_G = (abs(Delta_H3 - Delta_H5) + 2 * abs(Delta_H0 - Delta_H2) + abs(Delta_H6 - Delta_H8)) / 4

        return Delta_H_total, Dh_G

    def DeltaV(self, i, j):
        # 求垂直色差
        if self.mode == 0:
                    # 中 #
            Delta_V0 = self.DeltaVG(i-1, j)
            Delta_V1 = self.DeltaVRB(i, j)
            Delta_V2 = self.DeltaVG(i+1, j)
                    # 左 #
            Delta_V3 = self.DeltaVG(i-1, j-2)
            Delta_V4 = self.DeltaVRB(i, j-2)
            Delta_V5 = self.DeltaVG(i+1, j-2)
                    #右 #
            Delta_V6 = self.DeltaVG(i-1, j+2)
            Delta_V7 = self.DeltaVRB(i, j+2)
            Delta_V8 = self.DeltaVG(i+1, j+2)
        else:
                    # 中 #
            Delta_V0 = self.DeltaVRB(i-1, j)
            Delta_V1 = self.DeltaVG(i, j)
            Delta_V2 = self.DeltaVRB(i+1, j)
                    # 左 #
            Delta_V3 = self.DeltaVRB(i-1, j-2)
            Delta_V4 = self.DeltaVG(i, j-2)
            Delta_V5 = self.DeltaVRB(i+1, j-2)
                    #右 #
            Delta_V6 = self.DeltaVRB(i-1, j+2)
            Delta_V7 = self.DeltaVG(i, j+2)
            Delta_V8 = self.DeltaVRB(i+1, j+2)

        Delta_V1_total = (Delta_V0 + 2 * Delta_V1 + Delta_V2) / 4
        Delta_V2_total = (Delta_V3 + 2 * Delta_V4 + Delta_V5) / 4
        Delta_V3_total = (Delta_V6 + 2 * Delta_V7 + Delta_V8) / 4

        Delta_V_total = (Delta_V2_total + 2 * Delta_V1_total + Delta_V3_total) / 4
        Dv_G = (abs(Delta_V3 - Delta_V5) + 2 * abs(Delta_V0 - Delta_V2) + abs(Delta_V6 - Delta_V8)) / 4

        return Delta_V_total, Dv_G

    def __init__(self, oriImage):
        self.width = oriImage.shape[1]
        self.height = oriImage.shape[0]

        self.outputWidth = self.width
        self.outputHeight = self.height
        self.inputImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth, 3), dtype=np.uint8)
        self.mode = 0

    def Algorithm(self):
        inputImage = self.inputImage
        outImage = self.outputImage
        for i in range(self.outputHeight - 7):
            for j in range(5, self.outputWidth - 5):
                if i % 2 == 0: # even line B, G
                    if j % 2 == 0: # even pixel B
                        self.mode = 0
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

                        outImage[i, j, 1] = 255 if temp > 255 else 0 if temp < 0 else temp

                        # 補G #
                        # 補RB #
                            # 估計尚未插補的G值 #
                                # 左上 #
                        Delta_H0 = self.DeltaHRB(i-1, j-1)
                        Delta_V0 = self.DeltaVRB(i-1, j-1)
                                # 右上 #
                        Delta_H1 = self.DeltaHRB(i-1, j+1)
                        Delta_V1 = self.DeltaVRB(i-1, j+1)
                                # 左下 #
                        Delta_H2 = self.DeltaHRB(i+1, j-1)
                        Delta_V2 = self.DeltaVRB(i+1, j-1)
                                # 右下 #
                        Delta_H3 = self.DeltaHRB(i+1, j+1)
                        Delta_V3 = self.DeltaVRB(i+1, j+1)

                        Delta_H_total = (Delta_H0 + Delta_H1 + Delta_H2 + Delta_H3) / 4
                        Delta_V_total = (Delta_V0 + Delta_V1 + Delta_V2 + Delta_V3) / 4
                        Dh_O = (abs(Delta_H0 - Delta_H1) + abs(Delta_H2 - Delta_H3)) / 2
                        Dv_O = (abs(Delta_V0 - Delta_V2) + abs(Delta_V1 - Delta_V3)) / 2

                        if Dv_O <= Dh_O:
                            if Dv_O * 4 <= Dh_O:
                                temp = outImage[i, j, 1] - Delta_V_total
                            elif Dv_O * 2 <= Dh_O:
                                temp = outImage[i, j, 1] - (3 * Delta_V_total + Delta_H_total) / 4
                            else:
                                temp = outImage[i, j, 1] - (Delta_V_total + Delta_H_total) / 2
                        
                        else:
                        
                            if Dh_O * 4 <= Dv_O:
                                temp = outImage[i, j, 1] - Delta_H_total
                            elif Dh_O * 2 <= Dv_O:
                                temp = outImage[i, j, 1] - (3 * Delta_H_total + Delta_V_total) / 4
                            else:
                                temp = outImage[i, j, 1] - (Delta_H_total + Delta_V_total) / 2
                        

                        outImage[i, j, 0] = 255 if temp > 255 else 0 if temp < 0 else temp
                        outImage[i, j, 2] = inputImage[i, j]
                        # 補RB #
                    
                    else:           #  odd pixel G
                        self.mode = 1
                        # 補G 上 RB #
                            # 估計B的G值 #
                                # 左 #
                        Delta_H1 = self.DeltaHRB(i, j-1)
                        Delta_V1 = self.DeltaVRB(i, j-1)
                                # 右 #
                        Delta_H2 = self.DeltaHRB(i, j+1)
                        Delta_V2 = self.DeltaVRB(i, j+1)

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
                        
                        outImage[i, j, 2] = 255 if temp > 255 else 0 if temp < 0 else temp
                        
                            # 估計R的G值 #
                                # 上 #
                        GH1 = inputImage[i - 1, j] + ((inputImage[i - 1, j - 1] + inputImage[i - 1, j + 1]) / 2 - (inputImage[i - 1, j] * 2 + inputImage[i - 1, j - 2] + inputImage[i - 1, j + 2]) / 4)
                        GV1 = inputImage[i - 1, j] + ((inputImage[i, j] + inputImage[i - 2, j]) / 2 - (inputImage[i - 1, j] * 2 + inputImage[i + 1, j] + inputImage[i - 1, j]) / 4)
                        Delta_H1 = GH1 - inputImage[i - 1, j]
                        Delta_V1 = GV1 - inputImage[i - 1, j]
                                # 下 #
                        GH2 = inputImage[i + 1, j] + ((inputImage[i + 1, j - 1] + inputImage[i + 1, j + 1]) / 2 - (inputImage[i + 1, j] * 2 + inputImage[i + 1, j - 2] + inputImage[i + 1, j + 2]) / 4)
                        GV2 = inputImage[i + 1, j] + ((inputImage[i, j] + inputImage[i + 2, j]) / 2 - (inputImage[i + 1, j] * 2 + inputImage[i - 1, j] + inputImage[i + 1, j]) / 4)
                        Delta_H2 = GH2 - inputImage[i + 1, j]
                        Delta_V2 = GV2 - inputImage[i + 1, j]

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
                        
                        outImage[i, j, 0] = 255 if temp > 255 else 0 if temp < 0 else temp
                        outImage[i, j, 1] = inputImage[i, j]
                        # 補G 上 RB #
                    
                
                else:        # odd line G, R
                    
                    if j % 2 == 0: # even pixel G
                        self.mode = 1
                        # 補G 上 RB #
                            # 估計R的G值 #
                                # 左 #
                        GH1 = inputImage[i, j - 1] + ((inputImage[i, j - 2] + inputImage[i, j]) / 2 - (inputImage[i, j - 1] * 2 + inputImage[i, j - 3] + inputImage[i, j + 1]) / 4)
                        GV1 = inputImage[i, j - 1] + ((inputImage[i - 1, j - 1] + inputImage[i + 1, j - 1]) / 2 - (2 * inputImage[i, j - 1] + inputImage[i - 2, j - 1] + inputImage[i + 2, j - 1]) / 4)
                        Delta_H1 = GH1 - inputImage[i, j - 1]
                        Delta_V1 = GV1 - inputImage[i, j - 1]
                                # 右 #
                        GH2 = inputImage[i, j + 1] + ((inputImage[i, j] + inputImage[i, j + 2]) / 2 - (inputImage[i, j + 1] * 2 + inputImage[i, j - 1] + inputImage[i, j + 3]) / 4)
                        GV2 = inputImage[i, j + 1] + ((inputImage[i - 1, j + 1] + inputImage[i + 1, j + 1]) / 2 - (2 * inputImage[i, j + 1] + inputImage[i - 2, j + 1] + inputImage[i + 2, j + 1]) / 4)
                        Delta_H2 = GH2 - inputImage[i, j + 1]
                        Delta_V2 = GV2 - inputImage[i, j + 1]

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
                        
                        outImage[i, j, 0] = 255 if temp > 255 else 0 if temp < 0 else temp

                            # 估計B的G值 #
                                # 上 #
                        GH1 = inputImage[i - 1, j] + ((inputImage[i - 1, j - 1] + inputImage[i - 1, j + 1]) / 2 - (inputImage[i - 1, j] * 2 + inputImage[i - 1, j - 2] + inputImage[i - 1, j + 2]) / 4)
                        GV1 = inputImage[i - 1, j] + ((inputImage[i, j] + inputImage[i - 2, j]) / 2 - (inputImage[i - 1, j] * 2 + inputImage[i + 1, j] + inputImage[i - 1, j]) / 4)
                        Delta_H1 = GH1 - inputImage[i - 1, j]
                        Delta_V1 = GV1 - inputImage[i - 1, j]
                                # 下 #
                        GH2 = inputImage[i + 1, j] + ((inputImage[i + 1, j - 1] + inputImage[i + 1, j + 1]) / 2 - (inputImage[i + 1, j] * 2 + inputImage[i + 1, j - 2] + inputImage[i + 1 , j + 2]) / 4)
                        GV2 = inputImage[i + 1, j] + ((inputImage[i, j] + inputImage[i + 2, j]) / 2 - (inputImage[i + 1, j] * 2 + inputImage[i - 1, j] + inputImage[i + 1, j]) / 4)
                        Delta_H2 = GH2 - inputImage[i + 1, j]
                        Delta_V2 = GV2 - inputImage[i + 1, j]

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
                        

                        
                        outImage[i, j, 2] = 255 if temp > 255 else 0 if temp < 0 else temp
                        outImage[i, j, 1] = inputImage[i, j]
                        # 補G 上 RB #
                    
                    else:           #  odd pixel R
                        self.mode = 0
                        # 補G #
                        Delta_H_total, Dh_G = self.DeltaH(i, j)
                        Delta_V_total, Dv_G = self.DeltaV(i, j)

                            # 差補出G #
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
                        
                        outImage[i, j, 1] = 255 if temp > 255 else 0 if temp < 0 else temp
                        # 補G #
                        # 補RB #
                            # 估計尚未插補的G值 #
                                # 左上 #
                        Delta_H0 = self.DeltaHRB(i-1, j-1)
                        Delta_V0 = self.DeltaVRB(i-1, j-1)
                                # 右上 #
                        Delta_H1 = self.DeltaHRB(i-1, j+1)
                        Delta_V1 = self.DeltaVRB(i-1, j+1)
                                # 左下 #
                        Delta_H2 = self.DeltaHRB(i+1, j-1)
                        Delta_V2 = self.DeltaVRB(i+1, j-1)
                                # 右下 #
                        Delta_H3 = self.DeltaHRB(i+1, j+1)
                        Delta_V3 = self.DeltaVRB(i+1, j+1)

                        Delta_H_total = (Delta_H0 + Delta_H1 + Delta_H2 + Delta_H3) / 4
                        Delta_V_total = (Delta_V0 + Delta_V1 + Delta_V2 + Delta_V3) / 4

                        Dh_O = (abs(Delta_H0 - Delta_H1) + abs(Delta_H2 - Delta_H3)) / 2
                        Dv_O = (abs(Delta_V0 - Delta_V2) + abs(Delta_V1 - Delta_V3)) / 2

                        if Dv_O <= Dh_O:
                        
                            if Dv_O * 4 <= Dh_O:
                                temp = outImage[i, j, 1] - Delta_V_total
                            elif Dv_O * 2 <= Dh_O:
                                temp = outImage[i, j, 1] - (3 * Delta_V_total + Delta_H_total) / 4
                            else:
                                temp = outImage[i, j, 1] - (Delta_V_total + Delta_H_total) / 2
                        
                        else:
                        
                            if Dh_O * 4 <= Dv_O:
                                temp = outImage[i, j, 1] - Delta_H_total
                            elif Dh_O * 2 <= Dv_O:
                                temp = outImage[i, j, 1] - (3 * Delta_H_total + Delta_V_total) / 4
                            else:
                                temp = outImage[i, j, 1] - (Delta_H_total + Delta_V_total) / 2
                        

                        outImage[i, j, 2] = 255 if temp > 255 else 0 if temp < 0 else temp
                        outImage[i, j, 0] = inputImage[i, j]
                        # 補RB #
        return outImage

    