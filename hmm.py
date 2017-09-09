#-*- coding:utf-8
import utility
import numpy as np
class HMM:
    """
    Order 1 Hidden Markov Model

    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    """

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    """
       generate matrix from map

       Attributes
       ----------
        T : the length of the output

       Returns
       ----------
       hiddenStates,observationStates

    """
    def generateData(self,T):
        #根据分布列表，返回可能返回的Index
        def _getFromProbs(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        hiddenStates = np.zeros(T,dtype=int)
        observationsStates = np.zeros(T,dtype=int)
        hiddenStates[0] = _getFromProbs(self.pi) #产生第一个hiddenStates
        observationsStates[0] = _getFromProbs(self.B[hiddenStates[0]]) #产生第一个observationStates
        for t in range(1,T):
            hiddenStates[t] = _getFromProbs(self.A[hiddenStates[t-1]])
            observationsStates[t] = _getFromProbs((self.B[hiddenStates[t]]))

        return hiddenStates,observationsStates

    #维特比算法进行预测，即解码
    def viterbi(self,observationsSeq):
        T = len(observationsSeq)
        N = len(self.pi)
        prePath = np.zeros((T,N),dtype=int)
        dpMatrix = np.zeros((T,N),dtype=float)
        dpMatrix[0,:] = self.pi * self.B[:,observationsSeq[0]]

        for t in range(1,T):
            for n in range(N):
                probs = dpMatrix[t-1,:] * self.A[:,n] * self.B[n,observationsSeq[t]]
                prePath[t,n] = np.argmax(probs)
                dpMatrix[t,n] = np.max(probs)

        maxProb = np.max(dpMatrix[T-1,:])
        maxIndex = np.argmax(dpMatrix[T-1,:])
        path = [maxIndex]

        for t in reversed(range(1,T)):
            path.append(prePath[t,path[-1]])

        path.reverse()
        return maxProb,path

    #计算公式中的alpha二维数组
    def _forward(self,observationsSeq):
        T = len(observationsSeq)
        N = len(self.pi)
        alpha = np.zeros((T,N),dtype=float)
        alpha[0,:] = self.pi * self.B[:,observationsSeq[0]]  #numpy可以简化循环
        for t in range(1,T):
            for n in range(0,N):
                alpha[t,n] = np.dot(alpha[t-1,:],self.A[:,n]) * self.B[n,observationsSeq[t]] #使用内积简化代码
        return alpha

    #计算公式中的beita二维数组
    def _backward(self,observationsSeq):
        T = len(observationsSeq)
        N = len(self.pi)
        beta = np.zeros((T,N),dtype=float)
        beta[T-1,:] = 1
        for t in reversed(range(T-1)):
            for n in range(N):
                beta[t,n] = np.sum(self.A[n,:] * self.B[:,observationsSeq[t+1]] * beta[t+1,:])
        return beta

    #前后向算法学习参数
    def baumWelch(self,observationsSeq,criterion=0.001):
        T = len(observationsSeq)
        N = len(self.pi)

        while True:
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = self._forward(observationsSeq)

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = self._backward(observationsSeq)

            #根据公式求解XIt(i,j) = P(qt=Si,qt+1=Sj | O,λ)
            xi = np.zeros((T-1,N,N),dtype=float)
            for t in range(T-1):
                denominator = np.sum( np.dot(alpha[t,:],self.A) * self.B[:,observationsSeq[t+1]] * beta[t+1,:])
                for i in range(N):
                    molecular = alpha[t,i] * self.A[i,:] * self.B[:,observationsSeq[t+1]] * beta[t+1,:]
                    xi[t,i,:] = molecular / denominator

            #根据xi就可以求出gamma，注意最后缺了一项要单独补上来
            gamma = np.sum(xi,axis=2)
            prod = (alpha[T-1,:] * beta[T-1,:])
            gamma = np.vstack((gamma, prod /np.sum(prod)))

            newpi = gamma[0,:]
            newA = np.sum(xi,axis=0) / np.sum(gamma[:-1,:],axis=0).reshape(-1,1)
            newB = np.zeros(self.B.shape,dtype=float)

            for k in range(self.B.shape[1]):
                mask = observationsSeq == k
                newB[:,k] = np.sum(gamma[mask,:],axis=0) / np.sum(gamma,axis=0)

            if np.max(abs(self.pi - newpi)) < criterion and \
                                    np.max(abs(self.A - newA)) < criterion and \
                                    np.max(abs(self.B - newB)) < criterion:
                break

            self.A,self.B,self.pi = newA,newB,newpi


if __name__ == '__main__':
    hiddenStates = ("Healthy", "Fever")
    observationsStates = ("normal", "cold", "dizzy")
    pi = {"Healthy": 0.8, "Fever": 0.2}
    A = {
        "Healthy": {"Healthy": 0.6, "Fever": 0.4},
        "Fever": {"Healthy": 0.3, "Fever": 0.7}
    }
    B = {
        "Healthy": {"normal": 0.6, "cold": 0.2, "dizzy": 0.2},
        "Fever": {"normal": 0.1, "cold": 0.5, "dizzy": 0.4},
    }

    hStatesIndex = utility.generateStatesIndex(hiddenStates)
    oStatesIndex = utility.generateStatesIndex(observationsStates)
    A = utility.generateMatrix(A,hStatesIndex,hStatesIndex)
    B = utility.generateMatrix(B,hStatesIndex,oStatesIndex)
    pi = utility.generatePiVector(pi,hStatesIndex)
    h = HMM(A=A,B=B,pi=pi)

    states_data,observations_data = h.generateData(20)
    print states_data
    print observations_data

    # h.baumWelch(observations_data)
    # prob,path = h.viterbi(observations_data)
    # p = 0.0
    #
    # print path
    # print states_data
