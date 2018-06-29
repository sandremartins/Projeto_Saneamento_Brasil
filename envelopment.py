"""
Data Envelopment Analysis implementation

Sources:
Sherman & Zhu (2006) Service Productivity Management, Improving Service Performance using Data Envelopment Analysis (DEA) [Chapter 2]
ISBN: 978-0-387-33211-6
http://deazone.com/en/resources/tutorial

"""

import numpy as np
from scipy.optimize import fmin_slsqp


class DEA(object):

    def __init__(self, inputs, outputs):
        """
        Inicialize o objeto DEA com dados de inputs
        n = número de entidades (observações)
        m = número de inputs (variáveis, características)
        r = número de outputs
        :parâmetros inputs: inputs, n x m numpy array
        :parâmetros outputs: outputs, n x r numpy array
        :return: self
        """
                
        # dados fornecidos
        self.inputs = inputs
        self.outputs = outputs

        # parâmetros
        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]

        # iteradores
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)

        # arrays de resultado
        self.output_w = np.zeros((self.r, 1), dtype=np.float)  # pesos de output
        self.input_w = np.zeros((self.m, 1), dtype=np.float)  # pesos de input
        self.lambdas = np.zeros((self.n, 1), dtype=np.float)  # eficiências de unit
        self.efficiency = np.zeros_like(self.lambdas)  # thetas

        # nomes
        self.names = []

    def __efficiency(self, unit):
        """
        Função de eficiência com pesos já calculados
        :parâmetro unit: qual a unidade a ser computada
        :return: efficiency
        """

        # computa a efficiency
        denominator = np.dot(self.inputs, self.input_w)
        numerator = np.dot(self.outputs, self.output_w)

        return (numerator/denominator)[unit]

    def __target(self, x, unit):
        """
        Função alvo da teta para uma unidade
        :parâmetro x: pesos combinados
        :parâmetro unit: qual unidade de produção computar
        :return: theta
        """
        
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # desenrola os pesos
        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)

        return numerator/denominator

    def __constraints(self, x, unit):
        """
         Restrições para otimização de uma unidade
        :parâmetro x: pesos combinados
        :parâmetro unit: qual unidade de produção computar
        :return:  matrix constraints(matriz de restrição)
        """
            
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # desenrola os pesos
        constr = []  # inicie a matriz de restrição

        # para cada input, lambdas com inputs
        for input in self.input_:
            t = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t*self.inputs[unit, input] - lhs
            constr.append(cons)

        # para cada output, lambdas com outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)

        # para cada unit
        for u in self.unit_:
            constr.append(lambdas[u])

        return np.array(constr)

    def __optimize(self):
        """
        Otimização do modelo DEA
        Use: http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.linprog.html
        A = coeficientes nas restrições
        b = rhs de restrições
        c = coeficientes da função alvo
        :return:
        """
        
        d0 = self.m + self.r + self.n
        # iterar sobre units
        for unit in self.unit_:
            # pesos
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__target, x0, f_ieqcons=self.__constraints, args=(unit,))
            # desenrolar pesos
            self.input_w, self.output_w, self.lambdas = x0[:self.m], x0[self.m:(self.m+self.r)], x0[(self.m+self.r):]
            self.efficiency[unit] = self.__efficiency(unit)

    def name_units(self, names):
        """
        Forneça nomes para unidades para fins de apresentação
        :parâmetros names: uma lista de nomes, igual em comprimento ao número de unidades
        :return: nothing(nada)
        """
        assert(self.n == len(names))

        self.names = names

    def fit(self):
        """
        Otimize o conjunto de dados, gere a tabela básica
        :return: table
        """

        self.__optimize()  # optimize

        print("Thetas finais para cada unidade:\n")
        print("---------------------------\n")
        for n, eff in enumerate(self.efficiency):
            if len(self.names) > 0:
                name = "Theta unidade %s" % self.names[n]
            else:
                name = "Theta unidade %d" % (n+1)
            print("%s: %.4f" % (name, eff))
            print("\n")
        print("---------------------------\n")


if __name__ == "__main__":
    X = np.array([
        [20., 300.],
        [30., 200.],
        [40., 100.],
        [20., 200.],
        [10., 400.]
    ])
    y = np.array([
        [1000.],
        [1000.],
        [1000.],
        [1000.],
        [1000.]
    ])
    names = [
        'Bratislava',
        'Zilina',
        'Kosice',
        'Presov',
        'Poprad'
    ]
    dea = DEA(X,y)
    dea.name_units(names)
    dea.fit()
