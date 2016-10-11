import math
import numpy as np


class PerceptronClass():
    # Perceptron Class
    def __init__(self, x, w, txAprendizagem):
        self.x = x
        self.w = w
        self.txAprendizagem = txAprendizagem
        self.alfa = np.array([])

        # fase positiva
        self.Uv = np.array([])
        self.gUv = np.array([])
        self.PugUv = np.array([])
        self.Apos = np.array([])

        # fase negativa
        self.Uh_bar = np.array([])
        self.gUh_bar = np.array([])
        self.Uv_bar = np.array([])
        self.gUv_bar = np.array([])
        self.Aneg = np.array([])

        # erro
        self.erro = 0

    def inicia_alfa(self): # inciar após calcular Uv
        #dim = self.Uv.shape
        #self.alfa = np.random.random_sample(dim)
        self.alfa = np.array([
            [0.6, 0.9, 0.8],
            [0.6, 0.8, 0.8],
            [0.6, 0.7, 0.7],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]
        ])


    #fase positiva

    def Ph(self, h):
        r = 1
        print('+++++++++++++++++++++++')
        for i in range(len(self.w)):
            print(i)
            r *= 1 / (1 + math.pow(math.e, -self.w(i)))
        print('-----------------------')
        return r

    def calc_Uv(self):
        self.Uv =  np.dot(self.x, self.w)

    def calc_gUv(self):
        dim = self.Uv.shape
        self.gUv = np.zeros_like(self.Uv)
        for linha in range(dim[0]):
            for coluna in range(dim[1]):
                self.gUv[linha, coluna] = 1 / (1 + math.pow(math.e, -self.Uv[linha, coluna]))

    def calc_PugUv(self):
        self.PugUv = np.zeros_like(self.gUv)
        dim = self.PugUv.shape
        for linha in range(dim[0]):
            for coluna in range(dim[1]):
                if self.gUv[linha, coluna] >= self.alfa[linha, coluna]:
                    self.PugUv[linha, coluna] = 1
                else:
                    self.PugUv[linha, coluna] = 0


    # fase negativa

    def calc_Uh_bar(self):
        self.Uh_bar = np.dot(self.PugUv, np.transpose(self.w))

    def calc_gUh_bar(self):
        dim = self.Uh_bar.shape
        self.gUh_bar = np.zeros_like(self.Uh_bar)
        for linha in range(dim[0]):
            for coluna in range(dim[1]):
                self.gUh_bar[linha, coluna] = 1 / (1 + math.pow(math.e, -self.Uh_bar[linha, coluna]))
        self.gUh_bar[:,0] = 1

    def calc_Uv_bar(self):
        self.Uv_bar = np.dot(self.gUh_bar, self.w)

    def calc_gUv_bar(self):
        dim = self.Uv_bar.shape
        self.gUv_bar = np.zeros_like(self.Uv_bar)
        for linha in range(dim[0]):
            for coluna in range(dim[1]):
                self.gUv_bar[linha, coluna] = 1 / (1 + math.pow(math.e, -self.Uv_bar[linha, coluna]))

    # cálculo do erro e ajuste de pesos
    def calc_Apos(self):
        self.Apos = np.dot(np.transpose(self.x), self.gUv_bar) # x está com bias

    def calc_Aneg(self):
        self.Aneg = np.dot(np.transpose(self.gUh_bar), self.gUv_bar)

    def calc_erro(self):
        self.erro = np.sum(np.power(self.x - self.gUh_bar, 2))

    def ajustarPesos(self):
        dim = self.x.shape
        self.w += self.txAprendizagem * (self.Apos - self.Aneg) / dim[0]


    def epoca(self):
        # fase positiva
        self.calc_Uv()
        self.inicia_alfa()
        self.calc_gUv()
        self.calc_PugUv()

        # fase negativa
        self.calc_Uh_bar()
        self.calc_gUh_bar()
        self.calc_Uv_bar()
        self.calc_gUv_bar()

        # cálculo do erro
        self.calc_Apos()
        self.calc_Aneg()
        self.calc_erro()

        # ajuste de pesos
        self.ajustarPesos()