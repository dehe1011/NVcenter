import numpy as np
from qutip import sigmax, sigmay, sigmaz, Qobj, spre, sigmap, sigmam, sprepost, vector_to_operator, operator_to_vector, qeye
from qiskit import circuit, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
import scipy as sy
from scipy.spatial.transform import Rotation as R
from scipy.linalg import sqrtm
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import qutip as qt


def results2TT_matrix_import(res,n_timesteps):
    """ 
    Calculate the TT matrix as defined in the sumplementary material of "Single Qubit Error Mitigation by Simulating Non-Markovian Dynamics".

    Args:
        res(np.ndarray of shape(6,3,n_timesteps): result of the measurement of the expectationvalues of all 3 paulimatrices 
                                                for each eigenstate of the paulimatrices as initial state and different delay
                                                times. 1st dimenstion: Initial states (order xp,xm,yp,ym,z0,z1), 
                                                2nd dimension: pauli operator(x,y,z), 3rd dimension: different delay times
                                                
    Returns: 
        (np.ndarray of shape (4,4,n_timesteps): The TT matrix as defined in Equation(10) for each delay time

    """    
    

    timematrix = np.zeros((4,4,n_timesteps))
    timematrix[0,0,:] = 2 # the first column of TT is 0 except for the first entry
    
    for i in range(3):
        # For each entry i,0 the T matrix is given by the measurement of sigma_i with initial state in the first eigenstate of sigma_0 
        # plus the measurement of sigma_i with initial state in the second eigenstate of sigma_0
        timematrix[i+1,0,:] = res[4,i,:]+res[5,i,:]
        # for i,j>1: entries of TT are given by the 3x3 matrix T
        for j in range(3):
            # For each entry i,j the T matrix is given by the measurement of sigma_i with initial state in the + eigenstate of sigma_j 
            # minus the measurement of sigma_i with initial state in the minus eigenstate of sigma_j
            timematrix[i+1,j+1,:] = res[2*j,i,:]-res[2*j+1,i,:] 
    return timematrix/2



def TT_inverter(TT):
    """ 
    Calculate the inverse of the Matrix TT using the fact that it's of the special form as defined in the sumplementary material
    of "Single Qubit Error Mitigation by Simulating Non-Markovian Dynamics", equation (10).

    Args:
        TT (np.ndarray of shape (4,4)): TT matrix of a dynamical map

    Returns:
        (np.ndarray of shape (4,4)): TT matrix of the inverse map
        
    Implementation done by mirko
    """

                                                                                        
    TTinv = np.zeros((4,4))
    appo = np.zeros((3,3))
    appovec = np.zeros(3)
    for i in range(3):
        appovec[i] = TT[i+1,0]
        for j in range(3):
            appo[i,j] = TT[i+1,j+1]
    appoinv = np.linalg.inv(appo)
    vecinv = -np.dot(appoinv,appovec)
    
    TTinv[0,0] = 1
    for i in range(3):
        TTinv[i+1,0] = vecinv[i]
        for j in range(3):
            TTinv[i+1,j+1] = appoinv[i,j]
            
    return TTinv

def CPmap(TT,vec):
    """
    Apply  a linear qubit map in the 1, sigma_x,y,z basis (i.e. on the bloch sphere) to a vector in the same basis
    Args:
        TT (np.array of shape (4,4)):  a 4x4 matrix which represents a linear qubit map in the 1, sigma_x,y,z basis (i.e. on the bloch sphere)
        vec : 4 dimensional vector  
        
    Returns:
        (np.array of shape (2,2)):  the state after applying the linear map as a 2x2 matrix
    """             
    vec1 = np.dot(TT,vec)
    return vec1[0]*np.eye(2)+(vec1[1]*sigmax()+vec1[2]*sigmay()+vec1[3]*sigmaz()).full()



def ChoiMatrix(TT):
    """
    Calculates the Choi repersentation of the linear qubit map L which corresponds to TT (i.e. \sum_ij E_ij L(E_ij)).
    
    Args:
         TT (np.array(4,4)): A matrix which represents a linear qubit map in the 1, sigma_x,y,z basis (i.e. on the bloch sphere)
         
    Returns:
        #Output: The Choi repersentation of the linear qubit map L which corresponds to TT (i.e. \sum_ij E_ij L(E_ij))
    """               
    Choi=np.zeros([4,4],dtype = complex)
    Choi[0:2,0:2] = CPmap(TT,np.array([1/2,0,0,1/2]))
    Choi[2:5,0:2] = CPmap(TT,np.array([0,1/2,-1j/2,0]))
    Choi[0:2,2:5] = CPmap(TT,np.array([0,1/2,1j/2,0]))
    Choi[2:5,2:5] = CPmap(TT,np.array([1/2,0,0,-1/2]))
    return Choi

def PosAndNegChoi(Choi):
    #u, s, vh = np.linalg.svd(Choi)
    #ChoiP = (u *s.clip(min=0)) @ vh 
    #ChoiM = (u *s.clip(max=0)) @ vh 
    s, v = np.linalg.eig(Choi)
    AAp=0 
    AAm=0 
    for jj in range(0,4):
        #AAp += max(0,np.real(s[jj])) *np.outer(v[:,jj], np.conjugate(v[:,jj])) 
        #AAm += max(0,-np.real(s[jj])) *np.outer(v[:,jj], np.conjugate(v[:,jj]))  
        AAp += max(0,np.real(s[jj]))*np.outer(v[:,jj], np.conjugate(v[:,jj])) 
        AAm += max(0,-np.real(s[jj]))*np.outer(v[:,jj], np.conjugate(v[:,jj])) 
    #some condition for numerical stability
    if np.sum(np.abs(AAm)) != 0:
        for jj in range(0,4):
            AAp += 0.01 *np.outer(v[:,jj], np.conjugate(v[:,jj])) 
            AAm += 0.01 *np.outer(v[:,jj], np.conjugate(v[:,jj])) 
    return AAp, AAm

def ChoiToTmatrix(Choi):
                    #Input: Choi matrix representation of a linear qubit map
                    #Output: TT, a 4x4 matrix which represents a linear qubit map in the 1, sigma_x,y,z basis (i.e. on the bloch sphere)
    TT=np.zeros([4,4],dtype = complex)
    IdM = Qobj(Choi[0:2,0:2]+Choi[2:5,2:5])
    sigmaxM = Qobj(Choi[2:5,0:2]+Choi[0:2,2:5])
    sigmayM = Qobj(1j*Choi[2:5,0:2]-1j*Choi[0:2,2:5])
    sigmazM = Qobj(Choi[0:2,0:2] - Choi[2:5,2:5])
    TT = 1/2*np.array([[(IdM).tr(), (sigmaxM).tr(), (sigmayM).tr(), (sigmazM).tr()],\
                  [(sigmax()*IdM).tr(),(sigmax()*sigmaxM).tr(), (sigmax()*sigmayM).tr(), (sigmax()*sigmazM).tr()],\
                  [(sigmay()*IdM).tr(),(sigmay()*sigmaxM).tr(), (sigmay()*sigmayM).tr(), (sigmay()*sigmazM).tr()],\
                  [(sigmaz()*IdM).tr(),(sigmaz()*sigmaxM).tr(), (sigmaz()*sigmayM).tr(), (sigmaz()*sigmazM).tr()]])
    return TT


def TT_transformer(TT): #Collets your generation of variables    
    # determine the choi matrix of the map  
    Choi = ChoiMatrix(TT)
    # determine the Choi matrices of Lambda+ and Lambda-
    ChoiPlus,ChoiMinus = PosAndNegChoi(Choi)
    # determine lambda+, lambda- from choi matrices
    TTplus = ChoiToTmatrix(ChoiPlus)
    TTminus = ChoiToTmatrix(ChoiMinus)
    TTplusD = TTplus.transpose()
    TTminusD = TTminus.transpose()

    return TTplus, TTminus, TTplusD, TTminusD

def Tmatrix(Map):
                    #Input: a linear map acting on a qubit in the superoperator representation (i.e. as a 4x4 matrix using qutip functions)
                    #Output: TT a 4x4 matrix which represents a linear qubit map in the 1, sigma_x,y,z basis (i.e. on the bloch sphere)
    IdM = Qobj(vector_to_operator(Map*operator_to_vector(qeye(2))))
    sigmaxM = Qobj(vector_to_operator(Map*operator_to_vector(sigmax())))
    sigmayM = Qobj(vector_to_operator(Map*operator_to_vector(sigmay())))
    sigmazM = Qobj(vector_to_operator(Map*operator_to_vector(sigmaz())))
    TT = 1/2*np.array([[(IdM).tr(), (sigmaxM).tr(), (sigmayM).tr(), (sigmazM).tr()],\
                  [(sigmax()*IdM).tr(),(sigmax()*sigmaxM).tr(), (sigmax()*sigmayM).tr(), (sigmax()*sigmazM).tr()],\
                  [(sigmay()*IdM).tr(),(sigmay()*sigmaxM).tr(), (sigmay()*sigmayM).tr(), (sigmay()*sigmazM).tr()],\
                  [(sigmaz()*IdM).tr(),(sigmaz()*sigmaxM).tr(), (sigmaz()*sigmayM).tr(), (sigmaz()*sigmazM).tr()]])
    return TT


def D_Matrix(TT, epsilon = 0.01): #We have an input completely positive map L (represented by TTminusD) and we look for a matrix D
                                       # such that L^\dagger(1) + D^2 = (max eigenvalue(L^\dagger(1)) + epsilon)*1  
                    #Input: TT, a 4x4 matrix which represents a linear qubit map in the 1, sigma_x,y,z basis (i.e. on the bloch sphere)
                    #For our work, TT corresponds to the L_- in L = L_+ - L_-
                    #epsilon, some small number for numerical stability. 
                    #Output: Matt, a 4x4 matrix which represents the linear qubit map in the 1, sigma_x,y,z basis (i.e. on the bloch sphere)
                    #for the operation D \rho D
                    #max(AAA.eigenenergies()+epsilon), normalisation factor for the map
    AAA =CPmap(TT,np.array([1,0,0,0]))
    eigenenergies,_ = np.linalg.eig(AAA)
    D = Qobj(sy.linalg.sqrtm(max(eigenenergies+epsilon)*np.eye(2)-AAA))
    Matt = np.real(Tmatrix(sprepost(D,D)))
    return Matt, max(eigenenergies+epsilon)

def ExtremalMaps(TT):
    Choi = ChoiMatrix(TT)
    Amat = Choi[0:2,0:2]
    Amat1 = Choi[2:4,2:4]
    Choi1 = ChoiMatrix(TT)
    Choi2 = ChoiMatrix(TT)
    
    UU = np.dot(np.dot(sy.linalg.inv(sy.linalg.sqrtm(Amat)),Choi[0:2,2:4]),sy.linalg.inv(sy.linalg.sqrtm(Amat1)))
    u, s, vh = np.linalg.svd(UU)
    s =np.round(s,5)
    
    DiagMat1 = np.array([[np.exp(1j*np.arccos(s[0])),0],[0,np.exp(1j*np.arccos(s[1]))]])
    DiagMat2 = np.array([[np.exp(-1j*np.arccos(s[0])),0],[0,np.exp(-1j*np.arccos(s[1]))]])
    
    Choi1[0:2,2:4] =  np.dot(sy.linalg.sqrtm(Amat),np.dot(u,np.dot(DiagMat1,np.dot(vh,sy.linalg.sqrtm(Amat1)))))
    Choi1[2:4,0:2] =  np.dot(sy.linalg.sqrtm(Amat1),np.dot(np.conjugate(vh.transpose()),\
                                                           np.dot(DiagMat2,np.dot(np.conjugate(u.transpose()),sy.linalg.sqrtm(Amat)))))
    
    Choi2[0:2,2:4] =  np.dot(sy.linalg.sqrtm(Amat),np.dot(u,np.dot(DiagMat2,np.dot(vh,sy.linalg.sqrtm(Amat1)))))
    Choi2[2:4,0:2] =  np.dot(sy.linalg.sqrtm(Amat1),np.dot(np.conjugate(vh.transpose()),\
                                                           np.dot(DiagMat1,np.dot(np.conjugate(u.transpose()),sy.linalg.sqrtm(Amat)))))
    return [Choi1, Choi2]

def extr_maps(TTminusD,TTplusD): #Just a small collection of your functions preparing the Non-CP dynamical runs

    #Calculate D to make maps norm preserving
    # lamb is otherwise also called p
    Matt, lamb = D_Matrix(TTminusD,0.01)

    #Calculate the extremal maps in the Choi representation
    [ChoiM1, ChoiM2] = ExtremalMaps((TTminusD+Matt)/lamb)
    [ChoiP1, ChoiP2] = ExtremalMaps((TTplusD+Matt)/(1+lamb))

    return ChoiM1, ChoiM2, ChoiP1, ChoiP2, lamb

def AdjointChoiToChoi(AChoi):
                    #Input: Choi matrix repraboutesentation of a linear qubit map
                    #Output: Choi matrix representation of the adjoint of the linear qubit map (formula: eq. (25) of [1])            
    U23 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    Choi = np.conjugate(np.dot(U23,np.dot(AChoi,U23)))
    return Choi



def RotToUnitary(rot):
                    #Input: 3x3 rotation matrix representing a rotation on the Bloch sphere
                    #Output: 2x2 Unitary matrix representation of the rotation
    r = R.from_matrix(rot)
    v = r.as_rotvec()
    U = (-1j/2*(v[0]*sigmax()+v[1]*sigmay()+v[2]*sigmaz())).expm()
    return U

def onemax(val):
    if np.real(val) >= 1:
        return 0.999
    else:
        return val

def KrausOps(nu,mu):
                    #Input: 2 angles
                    #Output: 2 qubit Kraus operators (2x2 matrices) defined as eq. (18) in [1]
    Kraus1 = Qobj(np.array([[np.cos((mu-nu)/2),0],[0,np.cos((mu+nu)/2)]]))
    Kraus2 = Qobj(np.array([[0,np.sin((mu+nu)/2)],[np.sin((mu-nu)/2),0]]))
    return Kraus1, Kraus2, nu, mu


def KrausAndRotations(Choi):
    aaaa = ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,1:4]
    # calculate the singular value decomposition of lambda
    #ss: vector containing the diagonal entries
    uu, ss, vvh = np.linalg.svd(aaaa, full_matrices=True)
    # determine the unitaries from the rotation matrices
    U1 = RotToUnitary(np.linalg.det(uu)*uu)
    U2 = RotToUnitary(np.linalg.det(vvh)*vvh)
    #sigma = np.real(np.round(np.real(np.sin(np.arccos(np.linalg.det(uu)*np.linalg.det(vvh)*ss[0]))*np.sin(np.arccos(np.linalg.det(uu)*np.linalg.det(vvh)*ss[1])))\
    #         /np.dot(np.linalg.inv(np.linalg.det(uu)*uu),ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,0])[2]))
    #if np.real(np.dot(np.linalg.inv(np.linalg.det(uu)*uu),ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,0])[2])<0.0001:
    #    sigma = 1
    # new diagonal matrix is given by det(uu)*det(vvh)*ss  
    ang1,ang2,ang3 = np.linalg.det(uu)*np.linalg.det(vvh)*ss[0], np.linalg.det(uu)*np.linalg.det(vvh)*ss[0],   np.linalg.det(uu)*np.linalg.det(vvh)*ss[1]
    # make sure that the entries are smaller then one cause they are given by cos(angle)
    ang1,ang2,ang3 = onemax(ang1),onemax(ang2),onemax(ang3)
    
    # sign(sin(nu) * sin(mu))
    sigma = np.sign(np.real(np.sin(np.arccos(ang1))*np.sin(np.arccos(ang3))))\
            *np.sign(np.real(np.dot(np.linalg.inv(np.linalg.det(uu)*uu),ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,0])[2]))
    if sigma ==0:
        sigma = 1
    test = np.real(np.dot(np.linalg.inv(np.linalg.det(uu)*uu),ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,0])[2])
        
    theta1 = (sigma+1)/2*np.arccos(np.real(ang1)) + (-sigma+1)/2*(2*np.pi -np.arccos(np.real(ang2)))
    #theta1 = np.arccos(np.linalg.det(uu)*np.linalg.det(vvh)*ss[0]) + (-sigma+1)/2*(2*np.pi -np.arccos(np.linalg.det(uu)*np.linalg.det(vvh)*ss[0]))
    theta2 = np.arccos(np.real(ang3))
    Kraus1, Kraus2,nu, mu = KrausOps(theta1,theta2)
    
    #print('val1 = ', np.linalg.det(uu)*np.linalg.det(vvh)*ss[0],', val2 = ',np.linalg.det(uu)*np.linalg.det(vvh)*ss[0],', val3 = ',np.linalg.det(uu)*np.linalg.det(vvh)*ss[1])
    #print('ang1 = ',ang1,', ang2 = ',ang2,', ang3 = ',ang3)
    #print('theta1 = ',theta1,', theta2 = ',theta2)
    #print('theta1_1 = ',np.arccos(np.real(ang1)),', theta1_2 = ',-np.arccos(np.real(ang2)))
    #print('eccazzo: ', (sigma+1)/2*np.arccos(np.real(ang1)), ', eccazzo2: ', (-sigma+1)/2*(2*np.pi -np.arccos(np.real(ang2))))
    #print('sigma = ', sigma)
    
    return Kraus1, Kraus2, U1, U2, np.real(mu), np.real(nu), sigma


def KrausAndRotations_gate(Choi, gate_mat):
    aaaa = ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,1:4]
    # calculate the singular value decomposition of lambda
    #ss: vector containing the diagonal entries
    uu, ss, vvh = np.linalg.svd(aaaa, full_matrices=True)
    # determine the unitaries from the rotation matrices
    U1 = RotToUnitary(np.linalg.det(uu)*uu)
    U1 = gate_mat*U1
    U2 = RotToUnitary(np.linalg.det(vvh)*vvh)
    U2 = U2*gate_mat.dag()
    #sigma = np.real(np.round(np.real(np.sin(np.arccos(np.linalg.det(uu)*np.linalg.det(vvh)*ss[0]))*np.sin(np.arccos(np.linalg.det(uu)*np.linalg.det(vvh)*ss[1])))\
    #         /np.dot(np.linalg.inv(np.linalg.det(uu)*uu),ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,0])[2]))
    #if np.real(np.dot(np.linalg.inv(np.linalg.det(uu)*uu),ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,0])[2])<0.0001:
    #    sigma = 1
    # new diagonal matrix is given by det(uu)*det(vvh)*ss  
    ang1,ang2,ang3 = np.linalg.det(uu)*np.linalg.det(vvh)*ss[0], np.linalg.det(uu)*np.linalg.det(vvh)*ss[0],   np.linalg.det(uu)*np.linalg.det(vvh)*ss[1]
    # make sure that the entries are smaller then one cause they are given by cos(angle)
    ang1,ang2,ang3 = onemax(ang1),onemax(ang2),onemax(ang3)
    
    # sign(sin(nu) * sin(mu))
    sigma = np.sign(np.real(np.sin(np.arccos(ang1))*np.sin(np.arccos(ang3))))\
            *np.sign(np.real(np.dot(np.linalg.inv(np.linalg.det(uu)*uu),ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,0])[2]))
    if sigma ==0:
        sigma = 1
    test = np.real(np.dot(np.linalg.inv(np.linalg.det(uu)*uu),ChoiToTmatrix(AdjointChoiToChoi(Choi))[1:4,0])[2])
        
    theta1 = (sigma+1)/2*np.arccos(np.real(ang1)) + (-sigma+1)/2*(2*np.pi -np.arccos(np.real(ang2)))
    #theta1 = np.arccos(np.linalg.det(uu)*np.linalg.det(vvh)*ss[0]) + (-sigma+1)/2*(2*np.pi -np.arccos(np.linalg.det(uu)*np.linalg.det(vvh)*ss[0]))
    theta2 = np.arccos(np.real(ang3))
    Kraus1, Kraus2,nu, mu = KrausOps(theta1,theta2)
    
    #print('val1 = ', np.linalg.det(uu)*np.linalg.det(vvh)*ss[0],', val2 = ',np.linalg.det(uu)*np.linalg.det(vvh)*ss[0],', val3 = ',np.linalg.det(uu)*np.linalg.det(vvh)*ss[1])
    #print('ang1 = ',ang1,', ang2 = ',ang2,', ang3 = ',ang3)
    #print('theta1 = ',theta1,', theta2 = ',theta2)
    #print('theta1_1 = ',np.arccos(np.real(ang1)),', theta1_2 = ',-np.arccos(np.real(ang2)))
    #print('eccazzo: ', (sigma+1)/2*np.arccos(np.real(ang1)), ', eccazzo2: ', (-sigma+1)/2*(2*np.pi -np.arccos(np.real(ang2))))
    #print('sigma = ', sigma)
    
    return Kraus1, Kraus2, U1, U2, np.real(mu), np.real(nu), sigma


def QAlgo_creation(mu,nu,UP11,UP12,delay,xg='_Z'): #
    '''
    Takes the needed parameters  (mu, nu, UP11, UP12) and creates the quantum circuit that is subject to noise for a time delay and then applies       the gates defined by the parameters 
    Args
        mu, nu : parameters that determine the angles of the circuit
        UP11, UP12: Unitaries of the circuit
        delay: delay time [s] after which the map should be inverted
        xg: gate that creates the initial state       

    Returns
        quantum circuit to run on the ibm devices
    '''
    #Handling parameters
    alpha = (mu+nu)/2
    beta = (mu-nu)/2

    tgam1 = np.real(beta - alpha + np.pi/2)
    tgam2 = np.real(beta + alpha - np.pi/2)

    # We create the quantum circuit
    n = 2

    qs = QuantumRegister(n)
    cs = ClassicalRegister(n)
    qc = QuantumCircuit(qs, cs)

    delay_op = circuit.Delay(duration=delay, unit='s') 
    
    if xg == '_H':
        qc.h(0)
    elif xg == '_Z':
        qc.x([0])
    else:
        print("initialized in ground state")

    qc.append(delay_op,[0])
    qc.barrier()
    qc.unitary(UP12.full(),[0])
    qc.ry(tgam1,[1])
    qc.cx([0],[1])
    qc.ry(tgam2,[1])
    qc.measure([1], cs[0])
    qc.cx([1],[0])
    
    qc.unitary(UP11.full(),[0])            
    #qc.measure(s, c[1])
            
    return qc


def QAlgo_creation_gate(mu,nu,UP11,UP12,delay,xg='_Z', type_gate='H'): #Takes the needed parameters (mu, nu, UP11, UP12) and creates the quantum circuit to run

    #Handling parameters
    alpha = (mu+nu)/2
    beta = (mu-nu)/2

    tgam1 = np.real(beta - alpha + np.pi/2)
    tgam2 = np.real(beta + alpha - np.pi/2)

    # We create the quantum circuit
    n = 2

    qs = QuantumRegister(n)
    cs = ClassicalRegister(n)
    qc = QuantumCircuit(qs, cs)

    delay_op = circuit.Delay(duration=delay, unit='s') 
    
    if xg == '_H':
        qc.h(0)
    elif xg == '_Z':
        qc.x([0])
    else:
        print("initialized in ground state")

    qc.append(delay_op,[0])
    qc.barrier()
    if type_gate=='X':
        qc.x(0)
    else:    
        qc.h(0)
    qc.barrier()
    qc.unitary(UP12.full(),[0])
    qc.ry(tgam1,[1])
    qc.cx([0],[1])
    qc.ry(tgam2,[1])
    qc.measure([1], cs[0])
    qc.cx([1],[0])
    
    qc.unitary(UP11.full(),[0])     
    
    #qc.measure(s, c[1])
            
    return qc


def QAlgo_creation_gate_phase(mu,nu,UP11,UP12,delay,xg='_Z', type_gate='H', phase = 0): 
    '''
    Takes the needed parameters  (mu, nu, UP11, UP12) and creates the quantum circuit that is subject to noise for a time delay and then applies       the gates defined by the parameters. The waiting time is divided into two invervalls after which a phase is applied mimicing an external   
    magnetic field
    Args
        mu, nu : parameters that determine the angles of the circuit
        UP11, UP12: Unitaries of the circuit
        delay: delay time [s] after which the map should be inverted
        xg: gate that creates the initial state   
        type_gate: gate that is applied after the sensing and determines the measurement basis
        phase: phase that mimics an external magnetic field

    Returns
        quantum circuit to run on the ibm devices
    '''
    #Handling parameters
    alpha = (mu+nu)/2
    beta = (mu-nu)/2

    tgam1 = np.real(beta - alpha + np.pi/2)
    tgam2 = np.real(beta + alpha - np.pi/2)

    # We create the quantum circuit
    n = 2

    qs = QuantumRegister(n)
    cs = ClassicalRegister(n)
    qc = QuantumCircuit(qs, cs)

    delay_op = circuit.Delay(duration=delay/2, unit='s') 
   
    if xg == '_H':
        qc.h(0)
    elif xg == '_Z':
        qc.x([0])
    else:
        print("initialized in ground state")

    qc.append(delay_op,[0])
    qc.rz(phase,[0])
    qc.append(delay_op, [0])
    qc.barrier()
    if type_gate=='X':
        qc.x(0)
    else:    
        qc.h(0)
    qc.barrier()
    qc.unitary(UP12.full(),[0])
    qc.ry(tgam1,[1])
    qc.cx([0],[1])
    qc.ry(tgam2,[1])
    qc.measure([1], cs[0])
    qc.cx([1],[0])
    
    qc.unitary(UP11.full(),[0])     
    
    #qc.measure(s, c[1])
            
    return qc
    
def QAlgo_creation_gate_phase2(mu,nu,UP11,UP12,delay,xg='_Z', type_gate='H', delta_phi=2*np.pi/20, n_steps=1): 
    '''
    Takes the needed parameters  (mu, nu, UP11, UP12) and creates the quantum circuit that is subject to noise for a time delay and then applies       the gates defined by the parameters.
    The waiting time is divided into n+1 invervalls after which a phase is applied mimicing a stepwise applied external   
    magnetic field
    Args
        mu, nu : parameters that determine the angles of the circuit
        UP11, UP12: Unitaries of the circuit
        delay: delay time [s] after which the map should be inverted
        xg: gate that creates the initial state
        type_gate: gate that is applied after the sensing and determines the measurement basis
        delta_phi: phase that mimics an external magnetic field
        n_steps: amount of divisions of the waiting time

    Returns
        quantum circuit to run on the ibm devices
    '''

    #Handling parameters
    alpha = (mu+nu)/2
    beta = (mu-nu)/2

    tgam1 = np.real(beta - alpha + np.pi/2)
    tgam2 = np.real(beta + alpha - np.pi/2)

    # We create the quantum circuit
    n = 2

    qs = QuantumRegister(n)
    cs = ClassicalRegister(n)
    qc = QuantumCircuit(qs, cs)

    delay_op = circuit.Delay(duration=delay/(n_steps+1), unit='s') 
    if xg == '_H':
        qc.h(0)
    elif xg == '_Z':
        qc.x([0])
    else:
        print("initialized in ground state")

    qc.append(delay_op,[0])
    for i in range(n_steps):
        qc.rz(delta_phi,[0])
        qc.append(delay_op, [0])
    qc.barrier()
    if type_gate=='X':
        qc.x(0)
    else:    
        qc.h(0)
    qc.barrier()
    qc.unitary(UP12.full(),[0])
    qc.ry(tgam1,[1])
    qc.cx([0],[1])
    qc.ry(tgam2,[1])
    qc.measure([1], cs[0])
    qc.cx([1],[0])
    
    qc.unitary(UP11.full(),[0])     
    
    #qc.measure(s, c[1])
            
    return qc





def QAlgo_creation_qutip(mu,nu,UP11,UP12,rho_initial_c): 
    '''
    Takes the needed parameters  (mu, nu, UP11, UP12) and creates the qutip circuit that applies the according gates to it
    Args
        mu, nu : parameters that determine the angles of the circuit
        UP11, UP12: Unitaries of the circuit
        rho_initial_c: initial density matrix       

    Returns
        output states depending on the respective measurement result and according probabilities
    '''
    
    alpha = (mu+nu)/2
    beta = (mu-nu)/2
    c_not_upside_down= qt.Qobj(np.asarray([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]), dims=[[2, 2], [2, 2]])
    UP12 = qt.tensor(UP12, qt.qeye(2))
    UP11 = qt.tensor(UP11, qt.qeye(2))
    tgam1 = np.real(beta - alpha + np.pi/2)
    tgam2 = np.real(beta + alpha - np.pi/2)
    rho_initial = qt.Qobj(rho_initial_c)
    Z0, Z1 = qt.tensor(qt.qeye(2),qt.ket2dm(qt.basis(2, 0))),qt.tensor(qt.qeye(2), qt.ket2dm(qt.basis(2, 1)))
    

    # We create the quantum circuit
    initial_state_ancilla = qt.basis(2,0)
    #juhu
    initial_rho_ancilla = initial_state_ancilla*initial_state_ancilla.dag()
    initial_state = qt.tensor(rho_initial,initial_rho_ancilla)
    
    state = UP12*initial_state*UP12.dag()
    ry1= qt.tensor(qt.qeye(2),qt.gates.ry(tgam1))
    state =ry1*state*ry1.dag()
    state = qt.gates.cnot( )*state*qt.gates.cnot( ).dag()
    ry2= qt.tensor(qt.qeye(2),qt.gates.ry(tgam2))
    state = ry2*state*ry2.dag()
    #value, state = qt.measurement.measure(state, [Z0,Z1])
    #state =c_not_upside_down*state*c_not_upside_down.dag()
    states, probabilities = qt.measurement.measurement_statistics_povm(state, [Z0,Z1])
    #state =qutip_qip.operations.cnot(N=2,control = 1, target = 0 )*state*qutip_qip.operations.cnot(N=2,control = 0, target = 1 ).dag()
    state1 = c_not_upside_down*states[0]* c_not_upside_down.dag()
    state2 = c_not_upside_down*states[1]* c_not_upside_down.dag()
    #state = UP11*state*UP11.dag()
    state1 = UP11*state1*UP11.dag()
    state2 = UP11*state2*UP11.dag()
    return state1.ptrace(0),probabilities[0],state2.ptrace(0),probabilities[1]







def QAlgo_creation_comparison(delay,xg='_Z', type_gate='H'): #Takes the needed parameters (mu, nu, UP11, UP12) and creates the quantum circuit to run

   

    # We create the quantum circuit

    qc = QuantumCircuit(1, 1)

    delay_op = circuit.Delay(duration=delay, unit='s') 
    
    if xg == '_H':
        qc.h(0)
    elif xg == '_Z':
        qc.x([0])
    else:
        print("initialized in ground state")

    qc.append(delay_op,[0])
    qc.barrier()
    if type_gate == 'H':
        qc.h(0)
    elif type_gate== 'X':
        qc.x([0])
    else: 
        print("gate doesn't exist")
    return qc


def QAlgo_creation_comparison_phase(delay,xg='_Z', type_gate='H', phase=0): #Takes the needed parameters (mu, nu, UP11, UP12) and creates the quantum circuit to run

   

    # We create the quantum circuit

    qc = QuantumCircuit(1, 1)

    delay_op = circuit.Delay(duration=delay/2, unit='s') 

    if xg == '_H':
        qc.h(0)
    elif xg == '_Z':
        qc.x([0])
    else:
        print("initialized in ground state")

    qc.append(delay_op,[0])
    qc.rz(phase,[0])
    qc.append(delay_op, [0])
    qc.barrier()
    if type_gate == 'H':
        qc.h(0)
    elif type_gate== 'X':
        qc.x([0])
    else: 
        print("gate doesn't exist")
    return qc

def QAlgo_creation_comparison_phase2(delay,xg='_Z', type_gate='H', delta_phi=2*np.pi/20, n=1): #Takes the needed parameters (mu, nu, UP11, UP12) and creates the quantum circuit to run

   

    # We create the quantum circuit

    qc = QuantumCircuit(1, 1)

    delay_op = circuit.Delay(duration=delay/(n+1), unit='s') 
   
    if xg == '_H':
        qc.h(0)
    elif xg == '_Z':
        qc.x([0])
    else:
        print("initialized in ground state")

    qc.append(delay_op,[0])
    for i in range(n):
        qc.rz(delta_phi,[0])
        qc.append(delay_op, [0])
    qc.barrier()
    if type_gate == 'H':
        qc.h(0)
    elif type_gate== 'X':
        qc.x([0])
    else: 
        print("gate doesn't exist")
    return qc




def tomo_obs_setup(mult,layout):
    appo = [SparsePauliOp("II").apply_layout(layout),SparsePauliOp("IX").apply_layout(layout),SparsePauliOp("IY").apply_layout(layout),SparsePauliOp("IZ").apply_layout(layout),SparsePauliOp("XI").apply_layout(layout),SparsePauliOp("XX").apply_layout(layout),SparsePauliOp("XY").apply_layout(layout),SparsePauliOp("XZ").apply_layout(layout),SparsePauliOp("YI").apply_layout(layout),SparsePauliOp("YX").apply_layout(layout),SparsePauliOp("YY").apply_layout(layout),SparsePauliOp("YZ").apply_layout(layout),SparsePauliOp("ZI").apply_layout(layout),SparsePauliOp("ZX").apply_layout(layout),SparsePauliOp("ZY").apply_layout(layout),SparsePauliOp("ZZ").apply_layout(layout)]
    return [appo]*mult    

def obs_setup(mult,layout):
    """ 
    Create all 3 Pauli matrices and apply the defined layout to them. 
    Combine them in a list and create a list with repetitions of the list
    of Paulimatrices whith length coresponding to the amount of circuits that will be measured.

    Args:
        num_circuits (int): Number of circuits that are measured
        layout(Layout): Layout of the circuits (assumed to be the same for all circuits)

    Returns:
        list: List of length num_basis_states where each entry of the list is a list of the 3 Pauli matrices.
    """
    X_gate = SparsePauliOp("X").apply_layout(layout)
    Y_gate = SparsePauliOp("Y").apply_layout(layout)
    Z_gate = SparsePauliOp("Z").apply_layout(layout)
    appo = [[X_gate,Y_gate,Z_gate]]
    return appo*mult


def fid_T(T1,T2):
    
    appo = (np.matmul(T1.transpose(),T2).trace()/(np.sqrt(np.matmul(T1.transpose(),T1).trace())*np.sqrt(np.matmul(T2.transpose(),T2).trace())))
    
    return appo

def fid_meas(rho1,rho2):
    
    appo = sqrtm(np.matmul(np.matmul(sqrtm(rho1),rho2),sqrtm(rho1))).trace().real
    
    return appo**2

def fid_meas2(rho1,rho2):
    
    appo = fid_meas(rho1,rho2)
    
    return 1-(np.abs(1-appo)) 


def create_mitigated_circuit(xg,qubit,res,n_timesteps, delay_secs, backend, add_gate=False, adjust_map = False,type_gate = 'H', add_phase_gate = False, phase = 0):
    """
    Create a circuit that waits for some time and then restores the initial state based on the noise provile.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        res (np.nd array): noise map that was learnt for different waiting times
        n_timesteps (int): amount of times that the noise was learnt for
        delay_secs(np.array): waiting time for which the noise was learnt for 
        backend: backend for the passmanager
        add_gate(bool): specifies if a gate is added after the decay
        adjust_map(bool): specifies if the inverse noise map should be adjusted to the gate
        type_gate(String): gate that should be added
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state
        (list): list of lambda (weight factor between the positive and negative maps)
    """
    nshots = 10000
    qcs = []
    nshotss = []
    lambs = []
    
    layout = [qubit,qubit+1]
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, scheduling_method='alap', initial_layout=layout)
    
    obs_num = 16  #If below 'tomo_obs_setup' is used
    #obs_num = 9  #If below 'tomo_noId_obs_setup' isx used
    
    TT_tot = results2TT_matrix_import(res, n_timesteps) # ccalculate the dynamical map of the noise from the measurements
    for i in range(n_timesteps):
        TT = TT_inverter(TT_tot[:,:,i])
        TTplus, TTminus, TTplusD, TTminusD = TT_transformer(TT)
    
        ChoiM1, ChoiM2, ChoiP1, ChoiP2, lamb = extr_maps(TTminusD,TTplusD)
        lambs.append(lamb)
        if adjust_map:  
            if type_gate == 'H': 
                gate = qt.gates.snot()
            elif type_gate == 'X':
                gate = qt.sigmax()
            else:
                print("gate doesn't exist")
               
                gate =  qt.gates.snot()
            #Compute the 4 pairs of Kraus operators 
            KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations_gate(ChoiM1, gate)
            KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations_gate(ChoiM2, gate)
            KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations_gate(ChoiP1, gate)
            KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations_gate(ChoiP2, gate)
        else:
            KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations(ChoiM1)
            KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations(ChoiM2)
            KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations(ChoiP1)
            KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations(ChoiP2)
    
        sh1 = int((nshots/2)*lamb/(1+2*lamb))
        sh2 = int(nshots/2-sh1)
    
        delay = delay_secs[i]
        if add_gate:
            qc1 = QAlgo_creation_gate(muM1,nuM1,UM11,UM12,delay,xg=xg, type_gate= type_gate)
            qc2 = QAlgo_creation_gate(muM2,nuM2,UM21,UM22,delay,xg=xg,type_gate= type_gate)
            qc3 = QAlgo_creation_gate(muP1,nuP1,UP11,UP12,delay,xg=xg,type_gate= type_gate)
            qc4 = QAlgo_creation_gate(muP2,nuP2,UP21,UP22,delay,xg=xg,type_gate= type_gate)
        elif add_phase_gate:
            qc1 = QAlgo_creation_gate_phase(muM1,nuM1,UM11,UM12,delay,xg=xg, type_gate= type_gate, phase = phase)
            qc2 = QAlgo_creation_gate_phase(muM2,nuM2,UM21,UM22,delay,xg=xg,type_gate= type_gate, phase = phase)
            qc3 = QAlgo_creation_gate_phase(muP1,nuP1,UP11,UP12,delay,xg=xg,type_gate= type_gate, phase = phase)
            qc4 = QAlgo_creation_gate_phase(muP2,nuP2,UP21,UP22,delay,xg=xg,type_gate= type_gate, phase = phase)
        else:    
            qc1 = QAlgo_creation(muM1,nuM1,UM11,UM12,delay,xg=xg)
            qc2 = QAlgo_creation(muM2,nuM2,UM21,UM22,delay,xg=xg)
            qc3 = QAlgo_creation(muP1,nuP1,UP11,UP12,delay,xg=xg)
            qc4 = QAlgo_creation(muP2,nuP2,UP21,UP22,delay,xg=xg)
    
        qc1_isa = pm.run(qc1)
        qc2_isa = pm.run(qc2)
        qc3_isa = pm.run(qc3)
        qc4_isa = pm.run(qc4)
    
        qcs.append(qc1_isa)
        nshotss.append(sh1)
    
        qcs.append(qc2_isa)
        nshotss.append(sh1)
    
        qcs.append(qc3_isa)
        nshotss.append(sh2)
    
        qcs.append(qc4_isa)
        nshotss.append(sh2)
    
    obs_num = 16
    obs = tomo_obs_setup(len(qcs),qcs[0].layout)
    
    pub2 = list(zip(qcs, obs))
    return pub2, lambs




def create_mitigated_matrix_qutip(initial_states,res,n_timesteps,nshots_ges = 10000, sample_shots=False):

    """
    Create a circuit that waits for some time and then restores the initial state bplt.figure()
    plot_rho_diag(rho_map_circuit, n_timesteps=n_timesteps)
    plt.title('noise map')
    plt.ylim(-1,1.5)
    plt.legend()ased on the noise provile.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        res (np.nd array): noise map that was learnt for different waiting times
        n_timesteps (int): amount of times that the noise was learnt for
        delay_secs(np.array): waiting time for which the noise was learnt for 
        backend: backend for the passmanager
        add_gate(bool): specifies if a gate is added after the decay
        adjust_map(bool): specifies if the inverse noise map should be adjusted to the gate
        type_gate(String): gate that should be added
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state
        (list): list of lambda (weight factor between the positive and negative maps)
    """
    
    #qcs = []
    #nshotss = []
    nshots = nshots_ges/2
    lambs = []
    state_t=[]
 
    
    TT_tot = results2TT_matrix_import(res, n_timesteps) # calculate the dynamical map of the noise from the measurements
    for i in range(n_timesteps):
        TT = TT_inverter(TT_tot[:,:,i])
        TTplus, TTminus, TTplusD, TTminusD = TT_transformer(TT)
    
        ChoiM1, ChoiM2, ChoiP1, ChoiP2, lamb = extr_maps(TTminusD,TTplusD)
        lambs.append(lamb)
     
        KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations(ChoiM1)
        KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations(ChoiM2)
        KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations(ChoiP1)
        KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations(ChoiP2)

        state_11, p11, state_12, p12 = QAlgo_creation_qutip(muM1,nuM1,UM11,UM12, initial_states[i])
        state_21, p21, state_22, p22 = QAlgo_creation_qutip(muM2,nuM2,UM21,UM22, initial_states[i])
        state_31, p31, state_32, p32 = QAlgo_creation_qutip(muP1,nuP1,UP11,UP12,initial_states[i])
        state_41, p41, state_42, p42 = QAlgo_creation_qutip(muP2,nuP2,UP21,UP22,initial_states[i])
        if sample_shots:
            rng=np.random.default_rng(318)
            rng2=np.random.default_rng(567)
            rng3=np.random.default_rng(782)
            rng4=np.random.default_rng(113)
            sh1 = int(nshots*lamb/(1+2*lamb))
            sh2 = int(nshots-sh1)
            measurement1 = rng.random(sh1)
            n_11= np.sum(measurement1<p11)
            measurement2 = rng2.random(sh1) 
            n_21= np.sum(measurement2<p21)
            measurement3 = rng3.random(sh2) 
            n_31= np.sum(measurement3<p31)
            measurement4 = rng4.random(sh2) 
            n_41= np.sum(measurement4<p41)
            state1 = state_11*n_11/sh1+state_12*(1-n_11/sh1)
            state2 = state_21*n_21/sh1+state_22*(1-n_21/sh1)
            state3 = state_31*n_31/sh2+state_32*(1-n_31/sh2)
            state4 = state_41*n_41/sh2+state_42*(1-n_41/sh2)
        else:
            state1 = state_11*p11+state_12*p12
            state2 = state_21*p21+state_22*p22
            state3 = state_31*p31+state_32*p32
            state4 = state_41*p41+state_42*p42

        final_state = ((state3+state4)*(1+lamb)/2/(1+2*lamb)-(state1+state2)*lamb/2/(1+2*lamb))*(1+2*lamb)   
        #final_state = ((state_3/nshots+state_4/nshots)*sh2/nshots-(state_1/nshots+state_2/nshots)*sh1/nshots)*(1+2*lamb)   
    
        state_t.append(final_state)
    

    
    return state_t


def create_mitigated_value_qutip_var(initial_states,res,n_timesteps,nshots_ges = 10000,n_repetitions = 100, sample_shots=True, division_type = 0,add_gate = False, determine_sz = False, seed = 0, sample_second_measurement = False):

    """
    Create a circuit that waits for some time and then restores the initial state bplt.figure()
    plot_rho_diag(rho_map_circuit, n_timesteps=n_timesteps)
    plt.title('noise map')
    plt.ylim(-1,1.5)
    plt.legend()ased on the noise provile.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        res (np.nd array): noise map that was learnt for different waiting times
        n_timesteps (int): amount of times that the noise was learnt for
        delay_secs(np.array): waiting time for which the noise was learnt for 
        backend: backend for the passmanager
        add_gate(bool): specifies if a gate is added after the decay
        adjust_map(bool): specifies if the inverse noise map should be adjusted to the gate
        type_gate(String): gate that should be added
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state
        (list): list of lambda (weight factor between the positive and negative maps)
    """
    
    #qcs = []
    #nshotss = []
    nshots = nshots_ges/2
    lambs = []
    state_t=[]
    state_t2=[]
    var_t = []
 
    
    TT_tot = results2TT_matrix_import(res, n_timesteps) # calculate the dynamical map of the noise from the measurements
    for i in range(n_timesteps):
        TT = TT_inverter(TT_tot[:,:,i])
        TTplus, TTminus, TTplusD, TTminusD = TT_transformer(TT)
    
        ChoiM1, ChoiM2, ChoiP1, ChoiP2, lamb = extr_maps(TTminusD,TTplusD)
        lambs.append(lamb)
     
        KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations(ChoiM1)
        KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations(ChoiM2)
        KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations(ChoiP1)
        KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations(ChoiP2)
        if add_gate:
            gate = qt.Qobj(np.asarray([[ 1.-1.j,  1.+1.j],[-1.+1.j,  1.+1.j]])/2,dims = [2,2])
            UP11 = gate*UP11
            UM11 = gate*UM11
            UM21 = gate*UM21
            UP21 = gate*UP21

        state_11, p1, state_12, _ = QAlgo_creation_qutip(muM1,nuM1,UM11,UM12, initial_states[i])
        p11 = state_11[0,0]
        p12 = state_12[0,0]
        state_21, p2, state_22, _ = QAlgo_creation_qutip(muM2,nuM2,UM21,UM22, initial_states[i])
        p21 = state_21[0,0]
        p22 = state_22[0,0]
        state_31, p3, state_32, _ = QAlgo_creation_qutip(muP1,nuP1,UP11,UP12,initial_states[i])
        p31 = state_31[0,0]
        p32 = state_32[0,0]
        state_41, p4, state_42, _ = QAlgo_creation_qutip(muP2,nuP2,UP21,UP22,initial_states[i])
        p41 = state_41[0,0]
        p42 = state_42[0,0]
        
        
        rng=np.random.default_rng(318+seed+i)
        rng2=np.random.default_rng(567+seed+i)
        rng3=np.random.default_rng(782+seed+i)
        rng4=np.random.default_rng(113+seed+i)
        if division_type==0:
           
            sh1 =int(nshots*lamb/(1+2*lamb)) # int(nshots/2)#
            sh2 =int(nshots-sh1)# int(nshots/2)#0.5#int(nshots-sh1)
        elif division_type == 1:
            sh1 = int (nshots/2)
            sh2 = int(nshots-sh1)
        measurement1 = rng.random((2,sh1,n_repetitions))
        n_10= (np.sum((measurement1[0,:,:]<p1)*(measurement1[1,:,:]<p11),0)+np.sum((measurement1[0,:,:]>=p1)*(measurement1[1,:,:]<p12),0))/sh1
        measurement2 = rng2.random((2,sh1,n_repetitions)) 
        n_20= (np.sum((measurement2[0,:,:]<p2)*(measurement2[1,:,:]<p21),0)+np.sum((measurement2[0,:,:]>=p2)*(measurement2[1,:,:]<p22),0))/sh1
        measurement3 = rng3.random((2,sh2,n_repetitions)) 
        n_30= (np.sum((measurement3[0,:,:]<p3)*(measurement3[1,:,:]<p31),0)+np.sum((measurement3[0,:,:]>=p3)*(measurement3[1,:,:]<p32),0))/sh2
        measurement4 = rng4.random((2,sh2,n_repetitions)) 
        n_40= (np.sum((measurement4[0,:,:]<p4)*(measurement4[1,:,:]<p41),0)+np.sum((measurement4[0,:,:]>=p4)*(measurement4[1,:,:]<p42),0))/sh2
           
        if sample_second_measurement:
            n_10= (np.sum(p1*(measurement1[1,:,:]<p11),0)+np.sum((1-p1)*(measurement1[1,:,:]<p12),0))/sh1
            n_20= (np.sum(p2*(measurement2[1,:,:]<p21),0)+np.sum((1-p2)*(measurement2[1,:,:]<p22),0))/sh1
            n_30= (np.sum(p3*(measurement3[1,:,:]<p31),0)+np.sum((1-p3)*(measurement3[1,:,:]<p32),0))/sh2
            n_40= (np.sum(p4*(measurement4[1,:,:]<p41),0)+np.sum((1-p4)*(measurement4[1,:,:]<p42),0))/sh2
            
        final_res = ((n_30+n_40)*(1+lamb)/2/(1+2*lamb)-(n_10+n_20)*lamb/2/(1+2*lamb))*(1+2*lamb)
        if determine_sz:
            final_res = final_res-(1-final_res)
        var= np.var(final_res)
        avg = np.mean(final_res)
       
        #final_state = ((state_3/nshots+state_4/nshots)*sh2/nshots-(state_1/nshots+state_2/nshots)*sh1/nshots)*(1+2*lamb)   
    
        state_t.append(avg)
        var_t.append(var)
    

    
    return state_t, var_t,lambs


def create_mitigated_value_qutip(initial_states,res,n_timesteps,nshots_ges = 10000, sample_shots=True, division_type = 0, add_gate = False, seed = 0, sample_first_measurement = False ):

    """
    Create a circuit that waits for some time and then restores the initial state bplt.figure()
    plot_rho_diag(rho_map_circuit, n_timesteps=n_timesteps)
    plt.title('noise map')
    plt.ylim(-1,1.5)
    plt.legend()ased on the noise provile.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        res (np.nd array): noise map that was learnt for different waiting times
        n_timesteps (int): amount of times that the noise was learnt for
        delay_secs(np.array): waiting time for which the noise was learnt for 
        backend: backend for the passmanager
        add_gate(bool): specifies if a gate is added after the decay
        adjust_map(bool): specifies if the inverse noise map should be adjusted to the gate
        type_gate(String): gate that should be added
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state
        (list): list of lambda (weight factor between the positive and negative maps)
    """
    
    #qcs = []
    #nshotss = []
    nshots = nshots_ges/2
    lambs = []
    state_t=[]
    n_1_ar = []
    n_2_ar = []
    n_3_ar = []
    n_4_ar = []
 
    
    TT_tot = results2TT_matrix_import(res, n_timesteps) # calculate the dynamical map of the noise from the measurements
    for i in range(n_timesteps):
        TT = TT_inverter(TT_tot[:,:,i])
        TTplus, TTminus, TTplusD, TTminusD = TT_transformer(TT)
    
        ChoiM1, ChoiM2, ChoiP1, ChoiP2, lamb = extr_maps(TTminusD,TTplusD)
        lambs.append(lamb)
     
        KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations(ChoiM1)
        KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations(ChoiM2)
        KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations(ChoiP1)
        KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations(ChoiP2)

        if add_gate:
            gate = qt.Qobj(np.asarray([[ 1.-1.j,  1.+1.j],[-1.+1.j,  1.+1.j]])/2,dims = [2,2])
            UP11 = gate*UP11
            UM11 = gate*UM11
            UM21 = gate*UM21
            UP21 = gate*UP21
        
        state_11, p1, state_12, _ = QAlgo_creation_qutip(muM1,nuM1,UM11,UM12, initial_states[i])
        p11 = state_11[0,0]
        p12 = state_12[0,0]
        state_21, p2, state_22, _ = QAlgo_creation_qutip(muM2,nuM2,UM21,UM22, initial_states[i])
        p21 = state_21[0,0]
        p22 = state_22[0,0]
        state_31, p3, state_32, _ = QAlgo_creation_qutip(muP1,nuP1,UP11,UP12,initial_states[i])
        p31 = state_31[0,0]
        p32 = state_32[0,0]
        state_41, p4, state_42, _ = QAlgo_creation_qutip(muP2,nuP2,UP21,UP22,initial_states[i])
        p41 = state_41[0,0]
        p42 = state_42[0,0]
        
        if sample_shots:
            rng=np.random.default_rng(112+seed+i)#318)
            rng2=np.random.default_rng(334+seed+i)#567)
            rng3=np.random.default_rng(786+seed+i)#782)
            rng4=np.random.default_rng(943+seed+i)#113)
            if division_type==0:
                sh1 =int(nshots*lamb/(1+2*lamb))# int(nshots/2)#
                sh2 =int(nshots-sh1)# int(nshots/2)#0.5#int(nshots-sh1)
            elif division_type == 1:
                sh1 = int (nshots/2)
                sh2 = int(nshots-sh1)
            measurement1 = rng.random((2,sh1))
            n_10= (np.sum((measurement1[0,:]<p1)*(measurement1[1,:]<p11))+np.sum((measurement1[0,:]>=p1)*(measurement1[1,:]<p12)))/sh1
            measurement2 = rng2.random((2,sh1)) 
            n_20= (np.sum((measurement2[0,:]<p2)*(measurement2[1,:]<p21))+np.sum((measurement2[0,:]>=p2)*(measurement2[1,:]<p22)))/sh1
            measurement3 = rng3.random((2,sh2)) 
            n_30= (np.sum((measurement3[0,:]<p3)*(measurement3[1,:]<p31))+np.sum((measurement3[0,:]>=p3)*(measurement3[1,:]<p32)))/sh2
            measurement4 = rng4.random((2,sh2)) 
            n_40= (np.sum((measurement4[0,:]<p4)*(measurement4[1,:]<p41))+np.sum((measurement4[0,:]>=p4)*(measurement4[1,:]<p42)))/sh2
        elif sample_first_measurement:
            rng=np.random.default_rng(112+seed+i)#318)
            rng2=np.random.default_rng(334+seed+i)#567)
            rng3=np.random.default_rng(786+seed+i)#782)
            rng4=np.random.default_rng(943+seed+i)#113)
            if division_type==0:
                sh1 =int(nshots*lamb/(1+2*lamb))# int(nshots/2)#
                sh2 =int(nshots-sh1)# int(nshots/2)#0.5#int(nshots-sh1)
            elif division_type == 1:
                sh1 = int (nshots/2)
                sh2 = int(nshots-sh1)
            measurement1 = rng.random((2,sh1))
            n_10= (np.sum((measurement1[0,:]<p1)*p11)+np.sum((measurement1[0,:]>=p1)*p12))/sh1
            measurement2 = rng2.random((2,sh1)) 
            n_20= (np.sum((measurement2[0,:]<p2)*p21)+np.sum((measurement2[0,:]>=p2)*p22))/sh1
            measurement3 = rng3.random((2,sh2)) 
            n_30= (np.sum((measurement3[0,:]<p3)*p31)+np.sum((measurement3[0,:]>=p3)*p32))/sh2
            measurement4 = rng4.random((2,sh2)) 
            n_40= (np.sum((measurement4[0,:]<p4)*p41)+np.sum((measurement4[0,:]>=p4)*p42))/sh2
        else:
            n_10 = state_11[0,0]*p1+state_12[0,0]*(1-p1)
            n_20 = state_21[0,0]*p2+state_22[0,0]*(1-p2)
            n_30 = state_31[0,0]*p3+state_32[0,0]*(1-p3)
            n_40 = state_41[0,0]*p4+state_42[0,0]*(1-p4)
            
        final_res = ((n_30+n_40)*(1+lamb)/2/(1+2*lamb)-(n_10+n_20)*lamb/2/(1+2*lamb))*(1+2*lamb)   
        #final_state = ((state_3/nshots+state_4/nshots)*sh2/nshots-(state_1/nshots+state_2/nshots)*sh1/nshots)*(1+2*lamb)   
    
        state_t.append(final_res)
        n_1_ar.append(n_10)
        n_2_ar.append(n_20)
        n_3_ar.append(n_30)
        n_4_ar.append(n_40)
    

    
    return state_t



        
def restore_rho(results,lambs,n_timesteps):
    """ 
    Calculate the restored density matrix from the results of the measurement
    Args:
        results: result of the simulation on the simulator
        lambs(list): list of weights
        n_timesteps(int) : amount of times that were restored
    Returns:
        (list): density matrix for each recovered timestep as an array
        (list): density matrix of the qubit of interest for each recovered timestep as QObject
        (list): density matrix of the second qubit for each recovered timestep as QObject
    """ 
    siglist = [SparsePauliOp("II").to_matrix(),SparsePauliOp("IX").to_matrix(),SparsePauliOp("IY").to_matrix(),
               SparsePauliOp("IZ").to_matrix(),SparsePauliOp("XI").to_matrix(),SparsePauliOp("XX").to_matrix(),
               SparsePauliOp("XY").to_matrix(),SparsePauliOp("XZ").to_matrix(),SparsePauliOp("YI").to_matrix(),
               SparsePauliOp("YX").to_matrix(),SparsePauliOp("YY").to_matrix(),SparsePauliOp("YZ").to_matrix(),
               SparsePauliOp("ZI").to_matrix(),SparsePauliOp("ZX").to_matrix(),SparsePauliOp("ZY").to_matrix(),
               SparsePauliOp("ZZ").to_matrix()]
    rhoqcs = [np.zeros((4,4))]*(n_timesteps*4)
    nshots = 10000
    obs_num = 16
    for i in range(n_timesteps*4):
        for k in range(obs_num):
            rhoqcs[i] = rhoqcs[i] + (results[i].data.evs[k])*siglist[k]/4
    rhot = []
    for i in range(n_timesteps):
        sh1 = int((nshots/2)*lambs[i]/(1+2*lambs[i]))
        sh2 = int(nshots/2-sh1)
        #print(sh1, '   ', sh2)

        rhot1 = (rhoqcs[i*4]+rhoqcs[i*4+1])*sh1/nshots
        rhot2 = (rhoqcs[i*4+2]+rhoqcs[i*4+3])*sh2/nshots

        rhot.append(rhot2*(1+2*lambs[i]) - rhot1*(1+2*lambs[i]))
    twostates_corr = []
    state1_corr = []
    state2_corr = []
    for i in range(len(rhot)):
        twostates_corr.append(Qobj(rhot[i]))
        twostates_corr[i].dims = [[2,2],[2,2]]
        state1_corr.append(twostates_corr[i].ptrace(1).full())
        state2_corr.append(twostates_corr[i].ptrace(0).full())
    return rhot, state1_corr, state2_corr

def sample_rho(rho_list,shots):
    """
    Samples the up state of a density matrix with a finite amount of shots.
    Args
        rho_list: a list of density matrices
        shots: The amount of shots
    Returns:
        the sampled average of the density matrix being in the up state
    """    
        
    rng=np.random.default_rng(315338)
    measurement1 = rng.random((len(rho_list),shots))
    result=np.zeros(len(rho_list))
    for i in range(len(rho_list)):
        p=rho_list[i][0,0]
        n_i = np.sum(measurement1[i,:]<p)
        result[i]=n_i/shots
    return result

def sample_rho_var(rho_list,shots, n_repetitions = 100, determine_sz =False):
    """
    Samples the up state of a density matrix with a finite amount of shots.
    Args
        rho_list: a list of density matrices
        shots: The amount of shots
        n_repetitions: the amount of times the measurement is repeated to determine the variance
        
    Returns:
        the sampled average of the density matrix being in the up state
    """  
    rng=np.random.default_rng(315338)
    measurement1 = rng.random((len(rho_list),shots, n_repetitions))
    result=np.zeros(len(rho_list))
    var=np.zeros(len(rho_list))
    for i in range(len(rho_list)):
        p=rho_list[i][0,0]
        n_i = np.sum(measurement1[i,:,:]<p,0)/shots
        if determine_sz:
            n_i=n_i-(1-n_i)
        result[i]=np.mean(n_i)
        var[i] = np.var(n_i)
    return result, var
    
def noisy_rho(noise_record, gate_initial, n_timesteps):
    """
    calculate the timeevolution with noise of the system for a given initial state
    Args:
        noise_record(np.ndarray): learnt map of the noise
        gate_initial(string): gate that creates the initial state
        
    Returns:
        (list): density matrix of the system for each timestep
        np.array((2,2)): expected initial state
    """        
    red_siglist = [SparsePauliOp("I").to_matrix(),SparsePauliOp("X").to_matrix(),SparsePauliOp("Y").to_matrix(),SparsePauliOp("Z").to_matrix()]
    rhoqcs_noise = [np.zeros((2,2))]*(n_timesteps)
    
    
    for i in range(n_timesteps):
        if gate_initial == 'H':
            rhoqcs_noise[i] = 0.5*(red_siglist[0] + red_siglist[1]*noise_record[0,0,i] + red_siglist[2]*noise_record[0,1,i] 
                                   +red_siglist[3]*noise_record[0,2,i]) 
            initial_state = [[0.5,0.5],[0.5,0.5]]
        elif gate_initial == 'Z':
            rhoqcs_noise[i] =( 0.5*red_siglist[0] + 0.5*noise_record[5,0,i]*red_siglist[1] + 0.5*noise_record[5,1,i]*red_siglist[2]
                            +0.5*noise_record[5,2,i]*red_siglist[3])
            initial_state = [[0,0],[0,1]]
    return rhoqcs_noise,initial_state        


def create_noisy_circuit_gate(xg,qubit,n_timesteps, delay_secs, backend, type_gate = 'H',add_phase_gate = False, phase = 0):
    """
    Create a circuit that waits for some time and then applies a gate.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        n_timesteps (int): amount of times that the noise was learnt for
        delay_secs(np.array): waiting time for which the noise was learnt for 
        backend: backend for the passmanager
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state

    """
    qcs = []
    
    layout = [qubit]
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, scheduling_method='alap', initial_layout=layout)
    
    obs_num = 16  #If below 'tomo_obs_setup' is used
    #obs_num = 9  #If below 'tomo_noId_obs_setup' is used
    for i in range(n_timesteps):

        delay = delay_secs[i]
        if add_phase_gate:
            qc1 =  QAlgo_creation_comparison_phase(delay,xg=xg, type_gate=type_gate, phase = phase) 
        else:    
            qc1 =  QAlgo_creation_comparison(delay,xg=xg, type_gate=type_gate) 
        qc1_isa = pm.run(qc1)
        qcs.append(qc1_isa)
    
    obs_num = 16
    obs = obs_setup(len(qcs),qcs[0].layout)
    
    pub2 = list(zip(qcs, obs))
    return pub2
        

def restore_noisy_rho_gate(noise_record, n_timesteps):
    """
    calculate the timeevolution with noise of the system for a given initial state
    Args:
        noise_record(np.ndarray): learnt map of the noise
        gate_initial(string): gate that creates the initial state
        
    Returns:
        (list): density matrix of the system for each timestep
        np.array((2,2)): expected initial state
    """        
    red_siglist = [SparsePauliOp("I").to_matrix(),SparsePauliOp("X").to_matrix(),SparsePauliOp("Y").to_matrix(),SparsePauliOp("Z").to_matrix()]
    rhoqcs_noise = [np.zeros((2,2))]*(n_timesteps)
    
    
    for i in range(n_timesteps):
            rhoqcs_noise[i] = 0.5*(red_siglist[0] + red_siglist[1]*noise_record[i].data.evs[0]+ red_siglist[2]*noise_record[i].data.evs[1] 
                                   +red_siglist[3]*noise_record[i].data.evs[2])
    return rhoqcs_noise      



def create_mitigated_circuit_vary_phase(xg,qubit,res, delay_secs,phase_array, backend, t_ind=0, adjust_map = False,type_gate = 'H'):
    """
    Create a circuit that waits for some time , applies a phase gate,and then restores the initial state based on the noise provile.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        res (np.nd array): noise map that was learnt 
        delay_secs(np.nd array): times for which the noise was learnt
        
        phase_array: array_of_phases 
        backend: backend for the passmanager
        t_ind(int): index for the waiting time that should be used
        add_gate(bool): specifies if a gate is added after the decay
        adjust_map(bool): specifies if the inverse noise map should be adjusted to the gate
        type_gate(String): gate that should be added
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state
        (list): list of lambda (weight factor between the positive and negative maps)
    """
    nshots = 10000
    qcs = []
    nshotss = []
    lambs = []
    
    layout = [qubit,qubit+1]
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, scheduling_method='alap', initial_layout=layout)
    
    obs_num = 16  #If below 'tomo_obs_setup' is used
    #obs_num = 9  #If below 'tomo_noId_obs_setup' is used
    
    TT_tot = results2TT_matrix_import(res, len(delay_secs)) # ccalculate the dynamical map of the noise from the measurements
    
    TT = TT_inverter(TT_tot[:,:,t_ind])
    TTplus, TTminus, TTplusD, TTminusD = TT_transformer(TT)
    
    ChoiM1, ChoiM2, ChoiP1, ChoiP2, lamb = extr_maps(TTminusD,TTplusD)
    lambs.append(lamb)
    if adjust_map:  
        if type_gate =='H': 
            gate = qt.gates.snot()
        elif type_gate == 'X':
            gate = qt.sigmax()
        else:
            print("gate doesn't exist")
            gate =  qt.gates.snot()
        #Compute the 4 pairs of Kraus operators 
        KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations_gate(ChoiM1, gate)
        KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations_gate(ChoiM2, gate)            
        KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations_gate(ChoiP1, gate)
        KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations_gate(ChoiP2, gate)
    else:
        KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations(ChoiM1)
        KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations(ChoiM2)
        KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations(ChoiP1)
        KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations(ChoiP2)
    
    sh1 = int((nshots/2)*lamb/(1+2*lamb))
    sh2 = int(nshots/2-sh1)
    
    delay = delay_secs[t_ind]
      
    for i in range(len(phase_array)):
        qc1 = QAlgo_creation_gate_phase(muM1,nuM1,UM11,UM12,delay,xg=xg,type_gate= type_gate, phase = phase_array[i])
        qc2 = QAlgo_creation_gate_phase(muM2,nuM2,UM21,UM22,delay,xg=xg,type_gate= type_gate, phase = phase_array[i])
        qc3 = QAlgo_creation_gate_phase(muP1,nuP1,UP11,UP12,delay,xg=xg,type_gate= type_gate, phase = phase_array[i])
        qc4 = QAlgo_creation_gate_phase(muP2,nuP2,UP21,UP22,delay,xg=xg,type_gate= type_gate, phase = phase_array[i])

        qc1_isa = pm.run(qc1)
        qc2_isa = pm.run(qc2)
        qc3_isa = pm.run(qc3)
        qc4_isa = pm.run(qc4)
    
        qcs.append(qc1_isa)
        nshotss.append(sh1)
    
        qcs.append(qc2_isa)
        nshotss.append(sh1)
    
        qcs.append(qc3_isa)
        nshotss.append(sh2)
    
        qcs.append(qc4_isa)
        nshotss.append(sh2)
    
    obs_num = 16
    obs = tomo_obs_setup(len(qcs),qcs[0].lconstayout)
    
    pub2 = list(zip(qcs, obs))
    return pub2, lambs





def create_noisy_circuit_gate_vary_phase(xg,qubit, delay_secs, phase_array,backend,t_ind=0, type_gate = 'H'):
    """
    Create a circuit that waits for some time and then applies a gate.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        n_timesteps (int): amount of times that the noise was learnt for
        delay_secs(np.array): waiting time for which the noise was learnt for 
        backend: backend for the passmanager
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state

    """
    qcs = []
    
    layout = [qubit]
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, scheduling_method='alap', initial_layout=layout)
    
    obs_num = 16  #If below 'tomo_obs_setup' is used
    #obs_num = 9  #If below 'tomo_noId_obs_setup' is used
    delay = delay_secs[t_ind]
    for i in range(len(phase_array)):
        qc1 =  QAlgo_creation_comparison_phase(delay,xg=xg, type_gate=type_gate, phase = phase_array[i]) 
        qc1_isa = pm.run(qc1)
        qcs.append(qc1_isa)
    
    obs_num = 16
    obs = obs_setup(len(qcs),qcs[0].layout)
    
    pub2 = list(zip(qcs, obs))
    return pub2



def create_mitigated_circuit_vary_phase2(xg,qubit,res,n_timesteps, delay_secs, backend, add_gate=False, adjust_map = False,type_gate = 'H',  delta_phi = 2*np.pi/20):
    """
    Create a circuit that waits for some time and then restores the initial state based on the noise provile.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        res (np.nd array): noise map that was learnt for different waiting times
        n_timesteps (int): amount of times that the noise was learnt for
        delay_secs(np.array): waiting time for which the noise was learnt for 
        backend: backend for the passmanager
        add_gate(bool): specifies if a gate is added after the decay
        adjust_map(bool): specifies if the inverse noise map should be adjusted to the gate
        type_gate(String): gate that should be added
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state
        (list): list of lambda (weight factor between the positive and negative maps)
    """
    nshots = 10000
    qcs = []
    nshotss = []
    lambs = []
    
    layout = [qubit,qubit+1]

    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, scheduling_method='alap', initial_layout=layout)
    
    obs_num = 16  #If below 'tomo_obs_setup' is used
    #obs_num = 9  #If below 'tomo_noId_obs_setup' is used
    
    TT_tot = results2TT_matrix_import(res, n_timesteps) # ccalculate the dynamical map of the noise from the measurements
    for i in range(n_timesteps):
        TT = TT_inverter(TT_tot[:,:,i])
        TTplus, TTminus, TTplusD, TTminusD = TT_transformer(TT)
    
        ChoiM1, ChoiM2, ChoiP1, ChoiP2, lamb = extr_maps(TTminusD,TTplusD)
        lambs.append(lamb)
        if adjust_map:  
            if type_gate == 'H': 
                gate = qt.gates.snot()
            elif type_gate == 'X':
                gate = qt.sigmax()
            else:
                print("gate doesn't exist")
                gate =  qt.gates.snot()
            #Compute the 4 pairs of Kraus operators 
            KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations_gate(ChoiM1, gate)
            KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations_gate(ChoiM2, gate)
            KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations_gate(ChoiP1, gate)
            KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations_gate(ChoiP2, gate)
        else:
            KrausM11, KrausM12, UM11, UM12, muM1, nuM1, sigma1 = KrausAndRotations(ChoiM1)
            KrausM21, KrausM22, UM21, UM22, muM2, nuM2, sigma2 = KrausAndRotations(ChoiM2)
            KrausP11, KrausP12, UP11, UP12, muP1, nuP1, sigma3 = KrausAndRotations(ChoiP1)
            KrausP21, KrausP22, UP21, UP22, muP2, nuP2, sigma4 = KrausAndRotations(ChoiP2)
    
        sh1 = int((nshots/2)*lamb/(1+2*lamb))
        sh2 = int(nshots/2-sh1)
    
        delay = delay_secs[i]
       
        qc1 = QAlgo_creation_gate_phase2(muM1,nuM1,UM11,UM12,delay=delay,xg=xg, type_gate= type_gate, delta_phi = delta_phi, n_steps=i+1)
        qc2 = QAlgo_creation_gate_phase2(muM2,nuM2,UM21,UM22,delay=delay,xg=xg,type_gate= type_gate,  delta_phi = delta_phi, n_steps=i+1)
        qc3 = QAlgo_creation_gate_phase2(muP1,nuP1,UP11,UP12,delay=delay,xg=xg,type_gate= type_gate,  delta_phi = delta_phi, n_steps=i+1)
        qc4 = QAlgo_creation_gate_phase2(muP2,nuP2,UP21,UP22,delay=delay,xg=xg,type_gate= type_gate,  delta_phi = delta_phi, n_steps=i+1)
    
        qc1_isa = pm.run(qc1)
        qc2_isa = pm.run(qc2)
        qc3_isa = pm.run(qc3)
        qc4_isa = pm.run(qc4)
    
        qcs.append(qc1_isa)
        nshotss.append(sh1)
    
        qcs.append(qc2_isa)
        nshotss.append(sh1)
    
        qcs.append(qc3_isa)
        nshotss.append(sh2)
    
        qcs.append(qc4_isa)
        nshotss.append(sh2)
    
    obs_num = 16
    obs = tomo_obs_setup(len(qcs),qcs[0].layout)
    
    pub2 = list(zip(qcs, obs))
    return pub2, lambs
        

def create_noisy_circuit_gate_phase2(xg,qubit,n_timesteps, delay_secs, backend, type_gate = 'H', delta_phi=2*np.pi/20):
    """
    Create a circuit that waits for some time and then applies a gate.
    
    Args:
        xg (string): determines the gate used to initialize the state
        qubit (int): Number of the qubit of which the noise was learnt
        n_timesteps (int): amount of times that the noise was learnt for
        delay_secs(np.array): waiting time for which the noise was learnt for 
        backend: backend for the passmanager
    Returns:
        (list): list of circuits and observables that have to be measured to restore the initial state

    """
    qcs = []
    
    layout = [qubit]
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, scheduling_method='alap', initial_layout=layout)
    
    obs_num = 16  #If below 'tomo_obs_setup' is used
    #obs_num = 9  #If below 'tomo_noId_obs_setup' is used
    for i in range(n_timesteps):

        delay = delay_secs[i]
        qc1 =  QAlgo_creation_comparison_phase2(delay,xg=xg, type_gate=type_gate, delta_phi=delta_phi,n=i+1) 
     
        qc1_isa = pm.run(qc1)
        qcs.append(qc1_isa)
    
    obs_num = 16
    obs = obs_setup(len(qcs),qcs[0].layout)
    
    pub2 = list(zip(qcs, obs))
    return pub2



















