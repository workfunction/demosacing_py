import numpy as np
import numba as nb
from numba.experimental import jitclass

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
    ('Dv_G', nb.float64),
    ('Eh', nb.float64),
    ('Ev', nb.float64)
]

@jitclass(spec)
class MGEPI:
    def __init__(self, oriImage, scale=2):
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
        self.Eh = 0
        self.Ev = 0
        
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
                
    
    def _getDelta(self, dh, dv, gh, gv, eh=0, ev=0):
        s = gh + gv
        if 4*gh < gv:
            return dh
        elif 4*gv < gh:
            return dv
        else:
            if eh > 4*ev:
                return dh
            elif ev > 4*eh:
                return dv
            elif np.abs(dh - dv) > 20:
                return dh# if eh/(gh*gh) > ev/(gv*gv) else dv
            elif s == 0:
                return (dh + dv) / 2
            else:
                return (gh * dv + gv * dh) / s
        
    def __getDelta(self, dh, dv, gh, gv, eh=0, ev=0):
        s = gh + gv
        if s == 0:
            return (dh + dv) / 2
        else:
            return (gh * dv + gv * dh) / s
        
    def getDelta(self, dh, dv, gh, gv, eh=0, ev=0):        
        if gv <= gh:
            if gv * 4 <= gh:
                return dv
            elif ev > 4*eh:
                return dv
            elif gv * 2 <= gh:
                return (3 * dv + dh) / 4
            else:
                return (dv + dh) / 2
        else:
            if gh * 4 <= gv:
                return dh
            elif eh > 4*ev:
                return dh
            elif gh * 2 <= gv:
                return (3 * dh + dv) / 4
            else:
                return (dh + dv) / 2

    def _Delta(self, i, j):        
        Delta_H = np.zeros((3, 3))
        Delta_V = np.zeros((3, 3))
        diff = np.zeros((3, 3))
        det = np.zeros((3, 3))

        for m in range(0, 3):
            for n in range(0, 3):
                Delta_H[m, n] = self._DeltaH(i+2*(m-1), j+(n-1))        
                Delta_V[m, n] = self._DeltaV(i+(m-1), j+2*(n-1))
        
        #tmp = np.dot(Delta, np.array([0.05, 0.9, 0.05]))
        #Dh = np.dot(tmp, np.array([0.05, 0.9, 0.05]))
        
        #Gh = 1/(np.abs(self._DeltaH(i-1, j) - self._DeltaH(i+1, j))+1)
        Gh = np.dot(np.abs(Delta_H[:, 0] - Delta_H[:, 1]) + np.abs(Delta_H[:, 1] - Delta_H[:, 2]), np.array([0.25, 0.5, 0.25]))
        #Ghv = np.dot(np.abs(Delta_H[0, :] - Delta_H[1, :]) + np.abs(Delta_H[1, :] - Delta_H[2, :]), np.array([0.25, 0.5, 0.25]))
        Eh = 0#Ghv if Gh == 0 else Ghv / Gh
        
        #Gv = 1/(np.abs(self._DeltaV(i, j-1) - self._DeltaV(i, j+1))+1)
        #Gvh = np.dot(np.abs(Delta_V[:, 0] - Delta_V[:, 1]) + np.abs(Delta_V[:, 1] - Delta_V[:, 2]), np.array([0.25, 0.5, 0.25]))
        Gv = np.dot(np.abs(Delta_V[0, :] - Delta_V[1, :]) + np.abs(Delta_V[1, :] - Delta_V[2, :]), np.array([0.25, 0.5, 0.25]))
        Ev = 0#Gvh if Gv == 0 else Gvh / Gv  
        
        Gr = np.abs(Delta_V[1, :] - Delta_V[2, :])
        Gl = np.abs(Delta_V[0, :] - Delta_V[1, :])
        Gt = np.abs(Delta_H[:, 0] - Delta_H[:, 1])
        Gb = np.abs(Delta_H[:, 1] - Delta_H[:, 2])
        
        '''
        if i == 70 and j == 34:
            print(Delta)
            print(diff)
            print(Dh)
            print(Gh)
            print(Eh)
        '''        
        
        tmp_H = np.dot(np.array([0.05, 0.9, 0.05]), Delta_H)
        Dh = np.dot(tmp_H, np.array([0.05, 0.9, 0.05]))      
        tmp_V = np.dot(np.array([0.05, 0.9, 0.05]), Delta_V)
        Dv = np.dot(tmp_V, np.array([0.05, 0.9, 0.05]))  
        
        D = self.getDelta(Dh, Dv, Gh, Gv, Eh, Ev)
        
        #if i == 70 and j == 34:
        #    print(diff)        
        
        return D, Gh, Gv, Eh, Ev
    
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
            Delta_total, self.Dh_G, self.Dv_G, self.Eh, self.Ev = self._Delta(i, j)               
            
            '''
            if i == 70 and j == 34:
                self.Dh_G = 1
                self.Dv_G = 9
            ''' 
                # 插補出G #
            temp = inputImage[i, j] + Delta_total
            
            '''
            if i == 70 and j == 34:
                temp = 0
                print(temp)
            '''
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
            '''
            det_cH = np.zeros((2,2))
            det_cV = np.zeros((2,2))
            
            for m in range(2):
                for n in range(2):
                    det_cH[m, n] = self._DeltaH(i-1+m*2, i-1+n*2)
                    det_cV[m, n] = self._DeltaV(i-1+m*2, i-1+n*2)
                    
            Delta_H_total = np.sum(det_cH) / 4
            Delta_V_total = np.sum(det_cV) / 4
            
            Dh_O = np.sum(np.abs(det_cH[:, 0] - det_cH[:, 1])) / 2
            Dv_O = np.sum(np.abs(det_cV[0, :] - det_cV[1, :])) / 2
            '''
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
            
            Dh = abs(Delta_H1 - Delta_H2)
            
                # 估計R的G值 #
                    # 上 #
            Delta_H1 = self._DeltaH(i-1, j)
            Delta_V1 = self._DeltaV(i-1, j)
                    # 下 #
            Delta_H2 = self._DeltaH(i+1, j)
            Delta_V2 = self._DeltaV(i+1, j)
            
            Dv = abs(Delta_V1 - Delta_V2)

            Delta_H2_total = (Delta_H1 + Delta_H2) / 2
            Delta_V2_total = (Delta_V1 + Delta_V2) / 2

            det = self.getDelta(Delta_H1_total, Delta_V1_total, self.Dh_G, self.Dv_G)
            temp = inputImage[i, j] - det
            
            outImage[(2 if self.mode == 1 else 0)] = 255 if temp > 255 else 0 if temp < 0 else temp

            det = self.getDelta(Delta_H2_total, Delta_V2_total, self.Dh_G, self.Dv_G)
            temp = inputImage[i, j] - det
            
            outImage[(0 if self.mode == 1 else 2)] = 255 if temp > 255 else 0 if temp < 0 else temp
            outImage[1] = inputImage[i, j]
            # 補G 上 RB #
        
        s = self.Dh_G + self.Dv_G
        edge_factor = 0.5 if s == 0 else self.Dh_G / s

        '''
        if i == 70 and j == 34:
            print(self.inputImage[i-3:i+4, j-3:j+4])
            print(self.inputImage[i,j])
        '''
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
    
    def Demosaic(self):
        self.outputImage = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for oy in range(7, self.height - 7):
            for ox in range(5, self.width - 5):
                self.outputImage[oy, ox], e = self.mgbi(oy, ox)
        
        return self.outputImage

    def Algorithm(self):
        scale_factor_h = 0.5
        scale_factor_v = 0.5
        wsize = 5
        c = 2.5
        s = 2.5
        
        gamma = np.array([0.15, -0.75, 1, -0.3, -0.1])
        base_linear = np.array([0.10151382, -0.30232316, 1.0731945, 0.15881203, -0.031327657])
        base = np.array([0.13411383, -0.46307590, 1.2877634, 0.08989102, -0.048093690])
        
        tempWindow = np.zeros((wsize, wsize, 3), dtype=np.uint8)
        edge = np.zeros((wsize, wsize))
        for oy in range(self.outputHeight - 7):
            for ox in range(5, self.outputWidth - 5):

                if (wsize % 2 == 1):
                    j = int((ox)/2)
                    i = int((oy)/2)
                
                    for m in range(wsize):
                        for n in range(wsize):
                            tempWindow[m, n], edge[m, n] = self.mgbi(i-2+m, j-2+n)
                    
                    #odd = np.array([0.10151382, -0.30232316, 1.0731945, 0.15881203, -0.031327657])
                    #eve = np.array([-0.031327657, 0.15881203, 1.0731945, -0.30232316, 0.10151382])
                    
                    #odd = np.array([0, 0, 0.75,0.25, 0])
                    #eve = np.array([0, 0.25, 0.75, 0,0])
                    
                    e = edge[2, 2]#np.sum(edge[1:3, 1:3])/4
                      
                    b = base if (ox % 2) == 1 else np.flip(base)
                    a = base if (oy % 2) == 1 else np.flip(base)
                    
                    '''
                    if i == 70 and j == 34:
                        print(tempWindow[:, :, 1])
                        print(edge)
                    '''
                    
                    #b = b + gamma * 0.2 * (e-0.5)
                    #a = a + gamma * 0.2 * (0.5-e)
                    
                else:
                    x = (ox + 0.5) * (scale_factor_h) - 0.5
                    y = (oy + 0.5) * (scale_factor_v) - 0.5
                    
                    j = int(x)	#i = floor(x) 
                    i = int(y) #j = floor(y)
                    dx = x - float(j)
                    dy = y - float(i)
                    
                    for m in range(wsize):
                        for n in range(wsize):
                            tempWindow[m, n], edge[m, n] = self.mgbi(i-1+m, j-1+n)
                            
                    e = np.sum(edge[1:3, 1:3])/4
                
                    a = np.array(self.weight_cal(dy, c*(1-e)+s))
                    b = np.array(self.weight_cal(dx, c*(e)+s))
                        

                
                for color in range(3):
                    ori = tempWindow[:, :, color].astype(nb.float64)
                    temp = round(np.dot(np.dot(a, ori), b))
                    
                    temp *= (temp>0)
                    if temp > 255 :
                        temp = 255

                    self.outputImage[oy, ox, color] = temp

        return self.outputImage
