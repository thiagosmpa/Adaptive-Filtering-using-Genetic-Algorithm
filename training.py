import numpy as np
from numpy import random
import random as rd





class adaptiveFilt():
    def __init__ (self, x, d, w):
        self.x = x
        self.d = d
        self.w = w
        self.M = len(w)
        self.N = len(x)
        
            
    def matrixzation(self):
        j = 0
        training_matrix = np.zeros((self.N,self.M))
        
        for i in range(self.N-1,-1, -1):  
            for k in range(self.M):
                try:
                    training_matrix[j][k] = self.x[i+k]
                except IndexError:
                    continue
            j = j + 1
        return training_matrix
    
    
    
    
    
    def generate_population(self, popsize):
        
        population = -1 + 2*random.rand(popsize, self.M) 
        #random matrix of size popsize, M
        
        return population
    
    
    
    
    
    def piscinaCruzamento (self, popsize, population, x_m, sample):
        
        piscina = np.zeros((popsize,self.M))
        
        for aux in range(popsize):
            
            selected = 4
            randPop = np.zeros((selected, self.M))
            outs = np.zeros(selected)
            error = np.zeros(selected)
            
            selectIds = rd.sample(range(popsize), selected)
            
            for i in range(len(randPop)):
                
                randPop[i] = population[selectIds[i]]
                
                outs[i] = sum(randPop[i] * x_m[sample])
                error[i]  = outs[i] - self.d[sample]
                error[i] = np.square(error[i])
                
            
            sortError = np.sort(error)
            selectedAuxId = [i for i, x in enumerate(error) if x == sortError[0]]
            selectedAuxId = selectedAuxId[0]
            piscina[aux] = randPop[selectedAuxId]
            
    
        return piscina
    
    
    
    
    def crossover (self, population, popsize, piscinaCruz, crossOverProb):
        
        aux_population = np.zeros((popsize,self.M))
        
        child1 = np.zeros(self.M)
        child2 = np.zeros(self.M)
        w_child = np.zeros((self.M, 2))
        
        alfa = 0.2
        
        for i in range(0, popsize, 2):
            parent1 = piscinaCruz[i]
            parent2 = piscinaCruz[i+1]
            
            
            
            for j in range(self.M):
                w_child[j][0] = parent1[j]
                w_child[j][1] = parent2[j]
                w_child[j] = np.sort(w_child[j])
                
                child1[j] = rd.uniform(w_child[j][0] - alfa, w_child[j][1] + alfa)
                child2[j] = rd.uniform(w_child[j][0] - alfa, w_child[j][1] + alfa)
            
            aux_population[i] = child1
            aux_population[i+1] = child2
    
        return aux_population
    
    
    
    
    
    def mutatedPopulation(self, population, popsize, mutationProb):
        """"
        
    
        """
        numberOfGenesMutated = 3
        
        
        for i in range (popsize):
            prob = random.rand()
            
            if (prob < mutationProb):
                ngenesMutated = rd.randint(0,numberOfGenesMutated)
                # Define o numero de genes mutados de 0 a numberOfGenesMutated
                
                mutatedIndex = rd.sample(range(self.M), ngenesMutated)
                for j in range(self.M):
                    if j in mutatedIndex:
                        population[i][mutatedIndex] = rd.uniform(-1, 1)
                
        return population
    
    
    
    
    
    
    
    
    
class AdaptiveRun():
    
    def __init__ (self, x, d, w):
        self. x = x
        self.d = d
        self.w = w
        self.N = len(x)
        self.M = len(w)

    
    def run(self):
        """
        x: Input signal contaminated with noise.
        
        d: Desired signal.
        
        w: Filter coefficients.
        
        Return y, w
        """
        
    # =============================================================================
    #   INICIALIZAÇÃO DAS VARIÁVEIS
    # =============================================================================
        
        af = adaptiveFilt(self.x, self.d, self.w)
        x_m = af.matrixzation()      # input matrix with the size of M
        y = np.zeros(self.N)                     # output
        w_m = np.zeros((self.N,self.M))
        popsize = 10                        # population size
        
        
        # ngenerations = 100
        
        mutationProb = 0.1              # probabilidade de mutação
        crossOverProb = 0.9             # probabilidade de ocorrer o crossover
        
    # =============================================================================
    #   TESTES
    # =============================================================================
       
        if (self.M % 2 ) != 0:
            raise Exception ('Filter length must be even!')
        # filter length must be even to have the crossover 
    
        if (mutationProb > 1) or (mutationProb < 0):
            raise Exception ('Probability must be between 0 and 1 !')
        # mutation is percentage, so 0 < mutation < 1
        
        if (crossOverProb > 1) or (crossOverProb < 0):
            raise Exception ('Probability must be between 0 and 1 !')
        # crossOverProb is percentage, so 0 < crossOverProb < 1
        
        if (popsize % 2) != 0:
            raise Exception ('Popsize must be even')
        
    # =============================================================================
    #   GENERATING A THE FIRST POPULATION
    # =============================================================================
        
        population = af.generate_population(popsize)
        
    # =============================================================================
    #   VARRE O SINAL DE ENTRADA
    # =============================================================================
        
        for sample in range(self.N):
            print('Sample: ', sample)    
            meanSquareError= 9999
            minimumSquareError = 9999
            # meanSquareErrormust be high as possible to minimize 
            generation = 1    
           
            
           
            
           
            
    # =============================================================================
    #   DEFINE-SE O PONTO DE PARADA, DEPENDENDO DO NUMERO DE GERAÇÕES OU FIT
    # =============================================================================
            
            # for generation in range (ngenerations):
            while(minimumSquareError > 0.0001):    
                # run the iterations in a sample.    
                
                
                generation = generation + 1
                
    # =============================================================================
    #   CRIA-SE A PISCINA DE CRUZAMENTO COM SORTEIOS DA POPULAÇÃO INICIAL
    # =============================================================================
                
                piscinaCruz = af.piscinaCruzamento(popsize, population, x_m, sample)
    
    # =============================================================================
    #   ACONTECE O CROSSOVER
    # =============================================================================
                
                population = af.crossover(population, popsize, piscinaCruz, crossOverProb)
            
    # =============================================================================
    #   ACONTECE A MUTAÇÃO
    # =============================================================================
                
                population = af.mutatedPopulation(population, popsize, mutationProb)
                
    # =============================================================================
    #   OUTPUTS
    # =============================================================================
                
                error = np.zeros(popsize)
                y_w = np.zeros(popsize)
                for i in range(popsize):
                    y_w[i] = sum(population[i]*x_m[sample])
                    error[i] = y_w[i] - self.d[sample]
                    error[i] = np.square(error[i])
                
            
                meanSquareError= np.mean(error)
                
                # print(meanSquareError)
        
                best = np.sort(error)
                minimumSquareError = best[0]
                
                # Termina o While Error
            
            # print(generation)
            
            bestId = [i for i,x in enumerate(error) if x == best[0]]
            bestId = bestId[0]          #Em caso de valores repetidos, ele mostra todos 
                                        #os Ids, portanto, escolhe só o primeiro.
            w = population[bestId]
            w_m[sample] = w
            y[sample] = sum(x_m[sample]*w)
            
            erro = y[sample] - self.d[sample]
            
            # Terminal o For Sample
            
        return y, w_m


