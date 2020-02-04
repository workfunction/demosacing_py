from train.data import Data
from train.train import Train

def main():
    position_index = 0
    RGB_index = 1
    
    data = Data(position_index, RGB_index)
    data.save()
    
    dataload = Data()
    model = Train(e_num=20, data=dataload, index=position_index)
    
    modelload = Train()
    modelload.look()

if __name__ == "__main__":
    main()