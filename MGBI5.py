import numpy as np
import numba as nb

@nb.njit
def Clac(outputImage, width, height):
    for i in range(7, height - 7):
        for j in range(5, width - 5):
            if i % 2 == 0: # even line B, G
                if j % 2 == 0: # even pixel B
                    # 補G #
                        # 求水平色差 #
                            # 中 #
                    Delta_H0 = outputImage[i, j - 1, 1] - ((outputImage[i, j - 2, 2] + outputImage[i, j + 0, 2]) / 2 + (2 * outputImage[i, j - 1, 1] - outputImage[i, j - 3, 1] - outputImage[i, j + 1, 1]) / 4)
                    Delta_H1 = ((outputImage[i, j - 1, 1] + outputImage[i, j + 1, 1]) / 2 + (2 * outputImage[i, j + 0, 2] - outputImage[i, j - 2, 2] - outputImage[i, j + 2, 2]) / 4) - outputImage[i, j + 0, 2]
                    Delta_H2 = outputImage[i, j + 1, 1] - ((outputImage[i, j - 0, 2] + outputImage[i, j + 2, 2]) / 2 + (2 * outputImage[i, j + 1, 1] - outputImage[i, j - 1, 1] - outputImage[i, j + 3, 1]) / 4)
                    Delta_H1_total = (Delta_H0 + 2 * Delta_H1 + Delta_H2) / 4
                            # 上 #
                    Delta_H3 = outputImage[i - 2, j - 1, 1] - ((outputImage[i - 2, j - 2, 2] + outputImage[i - 2, j + 0, 2]) / 2 + (2 * outputImage[i - 2, j - 1, 1] - outputImage[i - 2, j - 3, 1] - outputImage[i - 2, j + 1, 1]) / 4)
                    Delta_H4 = ((outputImage[i - 2, j - 1, 1] + outputImage[i - 2, j + 1, 1]) / 2 + (2 * outputImage[i - 2, j + 0, 2] - outputImage[i - 2, j - 2, 2] - outputImage[i - 2, j + 2, 2]) / 4) - outputImage[i - 2, j + 0, 2]
                    Delta_H5 = outputImage[i - 2, j + 1, 1] - ((outputImage[i - 2, j - 0, 2] + outputImage[i - 2, j + 2, 2]) / 2 + (2 * outputImage[i - 2, j + 1, 1] - outputImage[i - 2, j - 1, 1] - outputImage[i - 2, j + 3, 1]) / 4)
                    Delta_H2_total = (Delta_H3 + 2 * Delta_H4 + Delta_H5) / 4
                            # 下 #
                    Delta_H6 = outputImage[i + 2, j - 1, 1] - ((outputImage[i + 2, j - 2, 2] + outputImage[i + 2, j + 0, 2]) / 2 + (2 * outputImage[i + 2, j - 1, 1] - outputImage[i + 2, j - 3, 1] - outputImage[i + 2, j + 1, 1]) / 4)
                    Delta_H7 = ((outputImage[i + 2, j - 1, 1] + outputImage[i + 2, j + 1, 1]) / 2 + (2 * outputImage[i + 2, j + 0, 2] - outputImage[i + 2, j - 2, 2] - outputImage[i + 2, j + 2, 2]) / 4) - outputImage[i + 2, j + 0, 2]
                    Delta_H8 = outputImage[i + 2, j + 1, 1] - ((outputImage[i + 2, j - 0, 2] + outputImage[i + 2, j + 2, 2]) / 2 + (2 * outputImage[i + 2, j + 1, 1] - outputImage[i + 2, j - 1, 1] - outputImage[i + 2, j + 3, 1]) / 4)
                    Delta_H3_total = (Delta_H6 + 2 * Delta_H7 + Delta_H8) / 4
                        # 求垂直色差
                            # 中 #
                    Delta_V0 = outputImage[i - 1, j, 1] - ((outputImage[i - 2, j, 2] + outputImage[i, j, 2]) / 2 + (2 * outputImage[i - 1, j, 1] - outputImage[i + 1, j, 1] - outputImage[i - 1, j, 1]) / 4)
                    Delta_V1 = ((outputImage[i - 1, j, 1] + outputImage[i + 1, j, 1]) / 2 + (2 * outputImage[i, j, 2] - outputImage[i - 2, j, 2] - outputImage[i + 2, j, 2]) / 4) - outputImage[i, j, 2]#中
                    Delta_V2 = outputImage[i + 1, j, 1] - ((outputImage[i, j, 2] + outputImage[i + 2, j, 2]) / 2 + (2 * outputImage[i + 1, j, 1] - outputImage[i - 1, j, 1] - outputImage[i + 1, j, 1]) / 4)
                    Delta_V1_total = (Delta_V0 + 2 * Delta_V1 + Delta_V2) / 4
                            # 左 #
                    Delta_V3 = outputImage[i - 1, j - 2, 1] - ((outputImage[i - 2, j - 2, 2] + outputImage[i, j - 2, 2]) / 2 + (2 * outputImage[i - 1, j - 2, 1] - outputImage[i + 1, j - 2, 1] - outputImage[i - 1, j - 2, 1]) / 4)
                    Delta_V4 = ((outputImage[i - 1, j - 2, 1] + outputImage[i + 1, j - 2, 1]) / 2 + (2 * outputImage[i, j - 2, 2] - outputImage[i - 2, j - 2, 2] - outputImage[i + 2, j - 2, 2]) / 4) - outputImage[i, j - 2, 2]#中
                    Delta_V5 = outputImage[i + 1, j - 2, 1] - ((outputImage[i, j - 2, 2] + outputImage[i + 2, j - 2, 2]) / 2 + (2 * outputImage[i + 1, j - 2, 1] - outputImage[i - 1, j - 2, 1] - outputImage[i + 1, j - 2, 1]) / 4)
                    Delta_V2_total = (Delta_V3 + 2 * Delta_V4 + Delta_V5) / 4
                            #右 #
                    Delta_V6 = outputImage[i - 1, j + 2, 1] - ((outputImage[i - 2, j + 2, 2] + outputImage[i, j + 2, 2]) / 2 + (2 * outputImage[i - 1, j + 2, 1] - outputImage[i + 1, j + 2, 1] - outputImage[i - 1, j + 2, 1]) / 4)
                    Delta_V7 = ((outputImage[i - 1, j + 2, 1] + outputImage[i + 1, j + 2, 1]) / 2 + (2 * outputImage[i, j + 2, 2] - outputImage[i - 2, j + 2, 2] - outputImage[i + 2, j + 2, 2]) / 4) - outputImage[i, j + 2, 2]#中
                    Delta_V8 = outputImage[i + 1, j + 2, 1] - ((outputImage[i, j + 2, 2] + outputImage[i + 2, j + 2, 2]) / 2 + (2 * outputImage[i + 1, j + 2, 1] - outputImage[i - 1, j + 2, 1] - outputImage[i + 1, j + 2, 1]) / 4)
                    Delta_V3_total = (Delta_V6 + 2 * Delta_V7 + Delta_V8) / 4
                            # 色差總和 #
                    Delta_H_total = (Delta_H2_total + 2 * Delta_H1_total + Delta_H3_total) / 4
                    Delta_V_total = (Delta_V2_total + 2 * Delta_V1_total + Delta_V3_total) / 4

                        # 求方向梯度 #
                    #Dh_G = (abs(outputImage[i-2, j-1, 1] - outputImage[i-2, j+1, 1]) + 2 * abs(outputImage[i, j-1, 1] - outputImage[i, j+1, 1]) + abs(outputImage[i+2, j-1, 1] - outputImage[i+2, j+1, 1])) / 4
                    #Dv_G = (abs(outputImage[i-1, j-2, 1] - outputImage[i+1, j-2, 1]) + 2 * abs(outputImage[i-1, j, 1] - outputImage[i+1, j, 1]) + abs(outputImage[i-1, j+2, 1] - outputImage[i+1, j+2, 1])) / 4
                    Dh_G = (abs(Delta_H3 - Delta_H5) + 2 * abs(Delta_H0 - Delta_H2) + abs(Delta_H6 - Delta_H8)) / 4
                    Dv_G = (abs(Delta_V3 - Delta_V5) + 2 * abs(Delta_V0 - Delta_V2) + abs(Delta_V6 - Delta_V8)) / 4
                        
                        # 插補出G #
                    if Dv_G <= Dh_G:
                        if Dv_G * 4 <= Dh_G:
                            outputImage[i, j, 1] = outputImage[i, j, 2] + Delta_V_total
                        elif Dv_G * 2 <= Dh_G:
                            outputImage[i, j, 1] = outputImage[i, j, 2] + (3 * Delta_V_total + Delta_H_total) / 4
                        else:
                            outputImage[i, j, 1] = outputImage[i, j, 2] + (Delta_V_total + Delta_H_total) / 2
                    else:
                        if Dh_G * 4 <= Dv_G:
                            outputImage[i, j, 1] = outputImage[i, j, 2] + Delta_H_total
                        elif Dh_G * 2 <= Dv_G:
                            outputImage[i, j, 1] = outputImage[i, j, 2] + (3 * Delta_H_total + Delta_V_total) / 4
                        else:
                            outputImage[i, j, 1] = outputImage[i, j, 2] + (Delta_H_total + Delta_V_total) / 2
                    
                    outputImage[i, j, 1] = 255 if outputImage[i, j, 1] > 255 else 0 if outputImage[i, j, 1] < 0 else outputImage[i, j, 1]

                    # 補G #
                    # 補RB #
                        # 估計尚未插補的G值 #
                            # 左上 #
                    GH1 = outputImage[i - 1, j - 1, 0] + ((outputImage[i - 1, j - 2, 1] + outputImage[i - 1, j, 1]) / 2 - (outputImage[i - 1, j - 1, 0] * 2 + outputImage[i - 1, j - 3, 0] + outputImage[i - 1, j + 1, 0]) / 4)
                    GV1 = outputImage[i - 1, j - 1, 0] + ((outputImage[i - 2, j - 1, 1] + outputImage[i, j - 1, 1]) / 2 - (outputImage[i - 1, j - 1, 0] * 2 + outputImage[i + 1, j - 1, 0] + outputImage[i - 1, j - 1, 0]) / 4)
                    Delta_H0 = GH1 - outputImage[i - 1, j - 1, 0]
                    Delta_V0 = GV1 - outputImage[i - 1, j - 1, 0]
                            # 右上 #
                    GH2 = outputImage[i - 1, j + 1, 0] + ((outputImage[i - 1, j, 1] + outputImage[i - 1, j + 2, 1]) / 2 - (outputImage[i - 1, j + 1, 0] * 2 + outputImage[i - 1, j - 1, 0] + outputImage[i - 1, j + 3, 0]) / 4)
                    GV2 = outputImage[i - 1, j + 1, 0] + ((outputImage[i, j + 1, 1] + outputImage[i - 2, j + 1, 1]) / 2 - (outputImage[i - 1, j + 1, 0] * 2 + outputImage[i + 1, j + 1, 0] + outputImage[i - 1, j + 1, 0]) / 4)
                    Delta_H1 = GH2 - outputImage[i - 1, j + 1, 0]
                    Delta_V1 = GV2 - outputImage[i - 1, j + 1, 0]
                            # 左下 #
                    GH3 = outputImage[i + 1, j - 1, 0] + ((outputImage[i + 1, j - 2, 1] + outputImage[i + 1, j, 1]) / 2 - (outputImage[i + 1, j - 1, 0] * 2 + outputImage[i + 1, j - 3, 0] + outputImage[i + 1, j + 1, 0]) / 4)
                    GV3 = outputImage[i + 1, j - 1, 0] + ((outputImage[i + 2, j - 1, 1] + outputImage[i, j - 1, 1]) / 2 - (outputImage[i + 1, j - 1, 0] * 2 + outputImage[i - 1, j - 1, 0] + outputImage[i + 1, j - 1, 0]) / 4)
                    Delta_H2 = GH3 - outputImage[i + 1, j - 1, 0]
                    Delta_V2 = GV3 - outputImage[i + 1, j - 1, 0]
                            # 右下 #
                    GH4 = outputImage[i + 1, j + 1, 0] + ((outputImage[i + 1, j, 1] + outputImage[i + 1, j + 2, 1]) / 2 - (outputImage[i + 1, j + 1, 0] * 2 + outputImage[i + 1, j - 1, 0] + outputImage[i + 1, j + 3, 0]) / 4)
                    GV4 = outputImage[i + 1, j + 1, 0] + ((outputImage[i, j + 1, 1] + outputImage[i + 2, j + 1, 1]) / 2 - (outputImage[i + 1, j + 1, 0] * 2 + outputImage[i - 1, j + 1, 0] + outputImage[i + 1, j + 1, 0]) / 4)
                    Delta_H3 = GH4 - outputImage[i + 1, j + 1, 0]
                    Delta_V3 = GV4 - outputImage[i + 1, j + 1, 0]

                    Delta_H_total = (Delta_H0 + Delta_H1 + Delta_H2 + Delta_H3) / 4
                    Delta_V_total = (Delta_V0 + Delta_V1 + Delta_V2 + Delta_V3) / 4
                    Dh_O = (abs(Delta_H0 - Delta_H1) + abs(Delta_H2 - Delta_H3)) / 2
                    Dv_O = (abs(Delta_V0 - Delta_V2) + abs(Delta_V1 - Delta_V3)) / 2

                    if Dv_O <= Dh_O:
                        if Dv_O * 4 <= Dh_O:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - Delta_V_total
                        elif Dv_O * 2 <= Dh_O:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (3 * Delta_V_total + Delta_H_total) / 4
                        else:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (Delta_V_total + Delta_H_total) / 2
                    
                    else:
                    
                        if Dh_O * 4 <= Dv_O:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - Delta_H_total
                        elif Dh_O * 2 <= Dv_O:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (3 * Delta_H_total + Delta_V_total) / 4
                        else:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (Delta_H_total + Delta_V_total) / 2
                    

                    outputImage[i, j, 0] = 255 if outputImage[i, j, 0] > 255 else 0 if outputImage[i, j, 0] < 0 else outputImage[i, j, 0]
                    # 補RB #
                
                else:           #  odd pixel G
                
                    # 補G 上 RB #
                        # 估計B的G值 #
                            # 左 #
                    GH1 = outputImage[i, j - 1, 2] + ((outputImage[i, j - 2, 1] + outputImage[i, j, 1]) / 2 - (outputImage[i, j - 1, 2] * 2 + outputImage[i, j - 3, 2] + outputImage[i, j + 1, 2]) / 4)
                    GV1 = outputImage[i, j - 1, 2] + ((outputImage[i - 1, j - 1, 1] + outputImage[i + 1, j - 1, 1]) / 2 - (2 * outputImage[i, j - 1, 2] + outputImage[i - 2, j - 1, 2] + outputImage[i + 2, j - 1, 2]) / 4)
                    Delta_H1 = GH1 - outputImage[i, j - 1, 2]
                    Delta_V1 = GV1 - outputImage[i, j - 1, 2]
                            # 右 #
                    GH2 = outputImage[i, j + 1, 2] + ((outputImage[i, j, 1] + outputImage[i, j + 2, 1]) / 2 - (outputImage[i, j + 1, 2] * 2 + outputImage[i, j - 1, 2] + outputImage[i, j + 3, 2]) / 4)
                    GV2 = outputImage[i, j + 1, 2] + ((outputImage[i - 1, j + 1, 1] + outputImage[i + 1, j + 1, 1]) / 2 - (2 * outputImage[i, j + 1, 2] + outputImage[i - 2, j + 1, 2] + outputImage[i + 2, j + 1, 2]) / 4)
                    Delta_H2 = GH2 - outputImage[i, j + 1, 2]
                    Delta_V2 = GV2 - outputImage[i, j + 1, 2]

                    Delta_H1_total = (Delta_H1 + Delta_H2) / 2
                    Delta_V1_total = (Delta_V1 + Delta_V2) / 2

                    if Dv_G <= Dh_G:
                    
                        if Dv_G * 4 <= Dh_G:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - Delta_V1_total
                        elif Dv_G * 2 <= Dh_G:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (3 * Delta_V1_total + Delta_H1_total) / 4
                        else:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (Delta_V1_total + Delta_H1_total) / 2
                    
                    else:
                    
                        if Dh_G * 4 <= Dv_G:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - Delta_H1_total
                        elif Dh_G * 2 <= Dv_G:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (3 * Delta_H1_total + Delta_V1_total) / 4
                        else:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (Delta_H1_total + Delta_V1_total) / 2
                    
                        # 估計R的G值 #
                            # 上 #
                    GH1 = outputImage[i - 1, j, 0] + ((outputImage[i - 1, j - 1, 1] + outputImage[i - 1, j + 1, 1]) / 2 - (outputImage[i - 1, j, 0] * 2 + outputImage[i - 1, j - 2, 0] + outputImage[i - 1, j + 2, 0]) / 4)
                    GV1 = outputImage[i - 1, j, 0] + ((outputImage[i, j, 1] + outputImage[i - 2, j, 1]) / 2 - (outputImage[i - 1, j, 0] * 2 + outputImage[i + 1, j, 0] + outputImage[i - 1, j, 0]) / 4)
                    Delta_H1 = GH1 - outputImage[i - 1, j, 0]
                    Delta_V1 = GV1 - outputImage[i - 1, j, 0]
                            # 下 #
                    GH2 = outputImage[i + 1, j, 0] + ((outputImage[i + 1, j - 1, 1] + outputImage[i + 1, j + 1, 1]) / 2 - (outputImage[i + 1, j, 0] * 2 + outputImage[i + 1, j - 2, 0] + outputImage[i + 1, j + 2, 0]) / 4)
                    GV2 = outputImage[i + 1, j, 0] + ((outputImage[i, j, 1] + outputImage[i + 2, j, 1]) / 2 - (outputImage[i + 1, j, 0] * 2 + outputImage[i - 1, j, 0] + outputImage[i + 1, j, 0]) / 4)
                    Delta_H2 = GH2 - outputImage[i + 1, j, 0]
                    Delta_V2 = GV2 - outputImage[i + 1, j, 0]

                    Delta_H1_total = (Delta_H1 + Delta_H2) / 2
                    Delta_V1_total = (Delta_V1 + Delta_V2) / 2

                    if Dv_G <= Dh_G:
                    
                        if Dv_G * 4 <= Dh_G:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - Delta_V1_total
                        elif Dv_G * 2 <= Dh_G:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (3 * Delta_V1_total + Delta_H1_total) / 4
                        else:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (Delta_V1_total + Delta_H1_total) / 2
                    
                    else:
                    
                        if Dh_G * 4 <= Dv_G:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - Delta_H1_total
                        elif Dh_G * 2 <= Dv_G:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (3 * Delta_H1_total + Delta_V1_total) / 4
                        else:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (Delta_H1_total + Delta_V1_total) / 2
                    

                    outputImage[i, j, 2] = 255 if outputImage[i, j, 2] > 255 else 0 if outputImage[i, j, 2] < 0 else outputImage[i, j, 2]
                    outputImage[i, j, 0] = 255 if outputImage[i, j, 0] > 255 else 0 if outputImage[i, j, 0] < 0 else outputImage[i, j, 0]
                    # 補G 上 RB #
                
            
            else:        # odd line G, R
            
                if j % 2 == 0: # even pixel G
                
                    # 補G 上 RB #
                        # 估計R的G值 #
                            # 左 #
                    GH1 = outputImage[i, j - 1, 0] + ((outputImage[i, j - 2, 1] + outputImage[i, j, 1]) / 2 - (outputImage[i, j - 1, 0] * 2 + outputImage[i, j - 3, 0] + outputImage[i, j + 1, 0]) / 4)
                    GV1 = outputImage[i, j - 1, 0] + ((outputImage[i - 1, j - 1, 1] + outputImage[i + 1, j - 1, 1]) / 2 - (2 * outputImage[i, j - 1, 0] + outputImage[i - 2, j - 1, 0] + outputImage[i + 2, j - 1, 0]) / 4)
                    Delta_H1 = GH1 - outputImage[i, j - 1, 0]
                    Delta_V1 = GV1 - outputImage[i, j - 1, 0]
                            # 右 #
                    GH2 = outputImage[i, j + 1, 0] + ((outputImage[i, j, 1] + outputImage[i, j + 2, 1]) / 2 - (outputImage[i, j + 1, 0] * 2 + outputImage[i, j - 1, 0] + outputImage[i, j + 3, 0]) / 4)
                    GV2 = outputImage[i, j + 1, 0] + ((outputImage[i - 1, j + 1, 1] + outputImage[i + 1, j + 1, 1]) / 2 - (2 * outputImage[i, j + 1, 0] + outputImage[i - 2, j + 1, 0] + outputImage[i + 2, j + 1, 0]) / 4)
                    Delta_H2 = GH2 - outputImage[i, j + 1, 0]
                    Delta_V2 = GV2 - outputImage[i, j + 1, 0]

                    Delta_H1_total = (Delta_H1 + Delta_H2) / 2
                    Delta_V1_total = (Delta_V1 + Delta_V2) / 2

                    if Dv_G <= Dh_G:
                    
                        if Dv_G * 4 <= Dh_G:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - Delta_V1_total
                        elif Dv_G * 2 <= Dh_G:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (3 * Delta_V1_total + Delta_H1_total) / 4
                        else:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (Delta_V1_total + Delta_H1_total) / 2
                    
                    else:
                    
                        if Dh_G * 4 <= Dv_G:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - Delta_H1_total
                        elif Dh_G * 2 <= Dv_G:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (3 * Delta_H1_total + Delta_V1_total) / 4
                        else:
                            outputImage[i, j, 0] = outputImage[i, j, 1] - (Delta_H1_total + Delta_V1_total) / 2
                    

                        # 估計B的G值 #
                            # 上 #
                    GH1 = outputImage[i - 1, j, 2] + ((outputImage[i - 1, j - 1, 1] + outputImage[i - 1, j + 1, 1]) / 2 - (outputImage[i - 1, j, 2] * 2 + outputImage[i - 1, j - 2, 2] + outputImage[i - 1, j + 2, 2]) / 4)
                    GV1 = outputImage[i - 1, j, 2] + ((outputImage[i, j, 1] + outputImage[i - 2, j, 1]) / 2 - (outputImage[i - 1, j, 2] * 2 + outputImage[i + 1, j, 2] + outputImage[i - 1, j, 2]) / 4)
                    Delta_H1 = GH1 - outputImage[i - 1, j, 2]
                    Delta_V1 = GV1 - outputImage[i - 1, j, 2]
                            # 下 #
                    GH2 = outputImage[i + 1, j, 2] + ((outputImage[i + 1, j - 1, 1] + outputImage[i + 1, j + 1, 1]) / 2 - (outputImage[i + 1, j, 2] * 2 + outputImage[i + 1, j - 2, 2] + outputImage[i + 1 , j + 2, 2]) / 4)
                    GV2 = outputImage[i + 1, j, 2] + ((outputImage[i, j, 1] + outputImage[i + 2, j, 1]) / 2 - (outputImage[i + 1, j, 2] * 2 + outputImage[i - 1, j, 2] + outputImage[i + 1, j, 2]) / 4)
                    Delta_H2 = GH2 - outputImage[i + 1, j, 2]
                    Delta_V2 = GV2 - outputImage[i + 1, j, 2]

                    Delta_H1_total = (Delta_H1 + Delta_H2) / 2
                    Delta_V1_total = (Delta_V1 + Delta_V2) / 2

                    if Dv_G <= Dh_G:
                    
                        if Dv_G * 4 <= Dh_G:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - Delta_V1_total
                        elif Dv_G * 2 <= Dh_G:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (3 * Delta_V1_total + Delta_H1_total) / 4
                        else:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (Delta_V1_total + Delta_H1_total) / 2
                    
                    else:
                    
                        if Dh_G * 4 <= Dv_G:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - Delta_H1_total
                        elif Dh_G * 2 <= Dv_G:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (3 * Delta_H1_total + Delta_V1_total) / 4
                        else:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (Delta_H1_total + Delta_V1_total) / 2
                    

                    
                    outputImage[i, j, 2] = 255 if outputImage[i, j, 2] > 255 else 0 if outputImage[i, j, 2] < 0 else outputImage[i, j, 2]
                    outputImage[i, j, 0] = 255 if outputImage[i, j, 0] > 255 else 0 if outputImage[i, j, 0] < 0 else outputImage[i, j, 0]
                    # 補G 上 RB #
                
                else:           #  odd pixel R
                
                    # 補G #
                        # 求水平色差 #
                            # 中 #
                    Delta_H0 = outputImage[i, j - 1, 1] - ((outputImage[i, j - 2, 0] + outputImage[i, j + 0, 0]) / 2 + (2 * outputImage[i, j - 1, 1] - outputImage[i, j - 3, 1] - outputImage[i, j + 1, 1]) / 4)#G-R
                    Delta_H1 = ((outputImage[i, j - 1, 1] + outputImage[i, j + 1, 1]) / 2 + (2 * outputImage[i, j + 0, 0] - outputImage[i, j - 2, 0] - outputImage[i, j + 2, 0]) / 4) - outputImage[i, j + 0, 0]#G-R
                    Delta_H2 = outputImage[i, j + 1, 1] - ((outputImage[i, j - 0, 0] + outputImage[i, j + 2, 0]) / 2 + (2 * outputImage[i, j + 1, 1] - outputImage[i, j - 1, 1] - outputImage[i, j + 3, 1]) / 4)#G-R
                    Delta_H1_total = (Delta_H0 + 2 * Delta_H1 + Delta_H2) / 4
                            # 上 #
                    Delta_H3 = outputImage[i - 2, j - 1, 1] - ((outputImage[i - 2, j - 2, 0] + outputImage[i - 2, j + 0, 0]) / 2 + (2 * outputImage[i - 2, j - 1, 1] - outputImage[i - 2, j - 3, 1] - outputImage[i - 2, j + 1, 1]) / 4)
                    Delta_H4 = ((outputImage[i - 2, j - 1, 1] + outputImage[i - 2, j + 1, 1]) / 2 + (2 * outputImage[i - 2, j + 0, 0] - outputImage[i - 2, j - 2, 0] - outputImage[i - 2, j + 2, 0]) / 4) - outputImage[i - 2, j + 0, 0]
                    Delta_H5 = outputImage[i - 2, j + 1, 1] - ((outputImage[i - 2, j - 0, 0] + outputImage[i - 2, j + 2, 0]) / 2 + (2 * outputImage[i - 2, j + 1, 1] - outputImage[i - 2, j - 1, 1] - outputImage[i - 2, j + 3, 1]) / 4)
                    Delta_H2_total = (Delta_H3 + 2 * Delta_H4 + Delta_H5) / 4
                            # 下 #
                    Delta_H6 = outputImage[i + 2, j - 1, 1] - ((outputImage[i + 2, j - 2, 0] + outputImage[i + 2, j + 0, 0]) / 2 + (2 * outputImage[i + 2, j - 1, 1] - outputImage[i + 2, j - 3, 1] - outputImage[i + 2, j + 1, 1]) / 4)
                    Delta_H7 = ((outputImage[i + 2, j - 1, 1] + outputImage[i + 2, j + 1, 1]) / 2 + (2 * outputImage[i + 2, j + 0, 0] - outputImage[i + 2, j - 2, 0] - outputImage[i + 2, j + 2, 0]) / 4) - outputImage[i + 2, j + 0, 0]
                    Delta_H8 = outputImage[i + 2, j + 1, 1] - ((outputImage[i + 2, j - 0, 0] + outputImage[i + 2, j + 2, 0]) / 2 + (2 * outputImage[i + 2, j + 1, 1] - outputImage[i + 2, j - 1, 1] - outputImage[i + 2, j + 3, 1]) / 4)
                    Delta_H3_total = (Delta_H6 + 2 * Delta_H7 + Delta_H8) / 4
                        # 求垂直色差 
                            # 中 #
                    Delta_V0 = outputImage[i - 1, j, 1] - ((outputImage[i - 2, j, 0] + outputImage[i, j, 0]) / 2 + (2 * outputImage[i - 1, j, 1] - outputImage[i + 1, j, 1] - outputImage[i - 1, j, 1]) / 4)
                    Delta_V1 = ((outputImage[i - 1, j, 1] + outputImage[i + 1, j, 1]) / 2 + (2 * outputImage[i, j, 0] - outputImage[i - 2, j, 0] - outputImage[i + 2, j, 0]) / 4) - outputImage[i, j, 0]#中
                    Delta_V2 = outputImage[i + 1, j, 1] - ((outputImage[i, j, 0] + outputImage[i + 2, j, 0]) / 2 + (2 * outputImage[i + 1, j, 1] - outputImage[i - 1, j, 1] - outputImage[i + 1, j, 1]) / 4)
                    Delta_V1_total = (Delta_V0 + 2 * Delta_V1 + Delta_V2) / 4
                            # 左 #
                    Delta_V3 = outputImage[i - 1, j - 2, 1] - ((outputImage[i - 2, j - 2, 0] + outputImage[i, j - 2, 0]) / 2 + (2 * outputImage[i - 1, j - 2, 1] - outputImage[i + 1, j - 2, 1] - outputImage[i - 1, j - 2, 1]) / 4)
                    Delta_V4 = ((outputImage[i - 1, j - 2, 1] + outputImage[i + 1, j - 2, 1]) / 2 + (2 * outputImage[i, j - 2, 0] - outputImage[i - 2, j - 2, 0] - outputImage[i + 2, j - 2, 0]) / 4) - outputImage[i, j - 2, 0]#中
                    Delta_V5 = outputImage[i + 1, j - 2, 1] - ((outputImage[i, j - 2, 0] + outputImage[i + 2, j - 2, 0]) / 2 + (2 * outputImage[i + 1, j - 2, 1] - outputImage[i - 1, j - 2, 1] - outputImage[i + 1, j - 2, 1]) / 4)
                    Delta_V2_total = (Delta_V3 + 2 * Delta_V4 + Delta_V5) / 4
                            #右 #
                    Delta_V6 = outputImage[i - 1, j + 2, 1] - ((outputImage[i - 2, j + 2, 0] + outputImage[i, j + 2, 0]) / 2 + (2 * outputImage[i - 1, j + 2, 1] - outputImage[i + 1, j + 2, 1] - outputImage[i - 1, j + 2, 1]) / 4)
                    Delta_V7 = ((outputImage[i - 1, j + 2, 1] + outputImage[i + 1, j + 2, 1]) / 2 + (2 * outputImage[i, j + 2, 0] - outputImage[i - 2, j + 2, 0] - outputImage[i + 2, j + 2, 0]) / 4) - outputImage[i, j + 2, 0]#中
                    Delta_V8 = outputImage[i + 1, j + 2, 1] - ((outputImage[i, j + 2, 0] + outputImage[i + 2, j + 2, 0]) / 2 + (2 * outputImage[i + 1, j + 2, 1] - outputImage[i - 1, j + 2, 1] - outputImage[i + 1, j + 2, 1]) / 4)
                    Delta_V3_total = (Delta_V6 + 2 * Delta_V7 + Delta_V8) / 4
                        # 總和 #
                    Delta_H_total = (Delta_H2_total + 2 * Delta_H1_total + Delta_H3_total) / 4
                    Delta_V_total = (Delta_V2_total + 2 * Delta_V1_total + Delta_V3_total) / 4
                        # 求方向梯度 #
                    Dh_G = (abs(Delta_H3 - Delta_H5) + 2 * abs(Delta_H0 - Delta_H2) + abs(Delta_H6 - Delta_H8)) / 4
                    Dv_G = (abs(Delta_V3 - Delta_V5) + 2 * abs(Delta_V0 - Delta_V2) + abs(Delta_V6 - Delta_V8)) / 4
                        # 差補出G #
                    if Dv_G <= Dh_G:
                    
                        if Dv_G * 4 <= Dh_G:
                            outputImage[i, j, 1] = outputImage[i, j, 0] + Delta_V_total
                        elif Dv_G * 2 <= Dh_G:
                            outputImage[i, j, 1] = outputImage[i, j, 0] + (3 * Delta_V_total + Delta_H_total) / 4
                        else:
                            outputImage[i, j, 1] = outputImage[i, j, 0] + (Delta_V_total + Delta_H_total) / 2
                    
                    else:
                    
                        if Dh_G * 4 <= Dv_G:
                            outputImage[i, j, 1] = outputImage[i, j, 0] + Delta_H_total
                        elif Dh_G * 2 <= Dv_G:
                            outputImage[i, j, 1] = outputImage[i, j, 0] + (3 * Delta_H_total + Delta_V_total) / 4
                        else:
                            outputImage[i, j, 1] = outputImage[i, j, 0] + (Delta_H_total + Delta_V_total) / 2
                    
                    outputImage[i, j, 0] = 255 if outputImage[i, j, 0] > 255 else 0 if outputImage[i, j, 0] < 0 else outputImage[i, j, 0]
                    # 補G #
                    # 補RB #
                        # 估計尚未插補的G值 #
                            # 左上 #
                    GH1 = outputImage[i - 1, j - 1, 2] + ((outputImage[i - 1, j - 2, 1] + outputImage[i - 1, j, 1]) / 2 - (outputImage[i - 1, j - 1, 2] * 2 + outputImage[i - 1, j - 3, 2] + outputImage[i - 1, j + 1, 2]) / 4)
                    GV1 = outputImage[i - 1, j - 1, 2] + ((outputImage[i - 2, j - 1, 1] + outputImage[i, j - 1, 1]) / 2 - (outputImage[i - 1, j - 1, 2] * 2 + outputImage[i + 1, j - 1, 2] + outputImage[i - 1, j - 1, 2]) / 4)
                    Delta_H0 = GH1 - outputImage[i - 1, j - 1, 2]
                    Delta_V0 = GV1 - outputImage[i - 1, j - 1, 2]
                            # 右上 #
                    GH2 = outputImage[i - 1, j + 1, 2] + ((outputImage[i - 1, j, 1] + outputImage[i - 1, j + 2, 1]) / 2 - (outputImage[i - 1, j + 1, 2] * 2 + outputImage[i - 1, j - 1, 2] + outputImage[i - 1, j + 3, 2]) / 4)
                    GV2 = outputImage[i - 1, j + 1, 2] + ((outputImage[i, j + 1, 1] + outputImage[i - 2, j + 1, 1]) / 2 - (outputImage[i - 1, j + 1, 2] * 2 + outputImage[i + 1, j + 1, 2] + outputImage[i - 1, j + 1, 2]) / 4)
                    Delta_H1 = GH2 - outputImage[i - 1, j + 1, 2]
                    Delta_V1 = GV2 - outputImage[i - 1, j + 1, 2]
                            # 左下 #
                    GH3 = outputImage[i + 1, j - 1, 2] + ((outputImage[i + 1, j - 2, 1] + outputImage[i + 1, j, 1]) / 2 - (outputImage[i + 1, j - 1, 2] * 2 + outputImage[i + 1, j - 3, 2] + outputImage[i + 1, j + 1, 2]) / 4)
                    GV3 = outputImage[i + 1, j - 1, 2] + ((outputImage[i + 2, j - 1, 1] + outputImage[i, j - 1, 1]) / 2 - (outputImage[i + 1, j - 1, 2] * 2 + outputImage[i - 1, j - 1, 2] + outputImage[i + 1, j - 1, 2]) / 4)
                    Delta_H2 = GH3 - outputImage[i + 1, j - 1, 2]
                    Delta_V2 = GV3 - outputImage[i + 1, j - 1, 2]
                            # 右下 #
                    GH4 = outputImage[i + 1, j + 1, 2] + ((outputImage[i + 1, j, 1] + outputImage[i + 1, j + 2, 1]) / 2 - (outputImage[i + 1, j + 1, 2] * 2 + outputImage[i + 1, j - 1, 2] + outputImage[i + 1, j + 3, 2]) / 4)
                    GV4 = outputImage[i + 1, j + 1, 2] + ((outputImage[i, j + 1, 1] + outputImage[i + 2, j + 1, 1]) / 2 - (outputImage[i + 1, j + 1, 2] * 2 + outputImage[i - 1, j + 1, 2] + outputImage[i + 1, j + 1, 2]) / 4)
                    Delta_H3 = GH4 - outputImage[i + 1, j + 1, 2]
                    Delta_V3 = GV4 - outputImage[i + 1, j + 1, 2]

                    Delta_H_total = (Delta_H0 + Delta_H1 + Delta_H2 + Delta_H3) / 4
                    Delta_V_total = (Delta_V0 + Delta_V1 + Delta_V2 + Delta_V3) / 4

                    Dh_O = (abs(Delta_H0 - Delta_H1) + abs(Delta_H2 - Delta_H3)) / 2
                    Dv_O = (abs(Delta_V0 - Delta_V2) + abs(Delta_V1 - Delta_V3)) / 2

                    if Dv_O <= Dh_O:
                    
                        if Dv_O * 4 <= Dh_O:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - Delta_V_total
                        elif Dv_O * 2 <= Dh_O:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (3 * Delta_V_total + Delta_H_total) / 4
                        else:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (Delta_V_total + Delta_H_total) / 2
                    
                    else:
                    
                        if Dh_O * 4 <= Dv_O:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - Delta_H_total
                        elif Dh_O * 2 <= Dv_O:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (3 * Delta_H_total + Delta_V_total) / 4
                        else:
                            outputImage[i, j, 2] = outputImage[i, j, 1] - (Delta_H_total + Delta_V_total) / 2
                    

                    outputImage[i, j, 2] = 255 if outputImage[i, j, 2] > 255 else 0 if outputImage[i, j, 2] < 0 else outputImage[i, j, 2]
                    # 補RB #
    return outputImage

class MGBI_5:
    def __init__(self, oriImage, width, height):
        self.width = width
        self.height = height

        self.outputWidth = width
        self.outputHeight = height
        self.outputImage = np.copy(oriImage)

    def Algorithm(self):

        return Clac(self.outputImage, self.width, self.height)